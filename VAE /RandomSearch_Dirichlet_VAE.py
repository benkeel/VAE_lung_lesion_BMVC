'''
This file is the script used to train the VAE on ARC (High Performance Computer)
'''

IMAGE_DIR = r"VAE_lung_lesion/Images"
results_path = r"/nobackup/mm17b2k/Dirichlet_VAE_RandomSearch"
save_results_path = r"/nobackup/mm17b2k/Dirichlet_VAE_RandomSearch/VAE_params.pt"

# Torch for data loading and for model buiding
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from torchsummary import summary
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.optim import RMSprop,Adam,SGD
from torch.distributions.dirichlet import Dirichlet
from torch.autograd import Variable

# Data preprocessing
import pandas as pd
from pandas.io.json import json_normalize
import json
import os
import re
from sklearn.model_selection import train_test_split

# Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats 
from PIL import Image
from pylab import rcParams

# Maths
import math
import numpy as np
from sklearn import metrics

# Other
from time import perf_counter
from collections import Counter,OrderedDict
import random
import warnings
import time

warnings.filterwarnings("ignore")

Run = np.load(results_path + '/' + 'run.npy', allow_pickle=True)#
Run = Run[0]
Run+=1
print("run:", Run)

# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

all_files_list = [f for f in os.listdir(IMAGE_DIR)]
all_files_list.sort()
        
#print(len(all_files_list))
#print(all_files_list[0:10])

# settings for reproducibility
torch.manual_seed(int(time.time()))
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# misc
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create class to load image data from files
class LoadImages(Dataset):
    def __init__(self, main_dir, files_list, HU_Upper, HU_Lower):
        # Set the loading directory
        self.main_dir = main_dir

        # Transforms
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Get list of all image file names
        self.all_imgs = files_list
        
        # Get HU limits
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower
                  
        
    def __len__(self):
        # Return the previously computed number of images
        return len(self.all_imgs)    

    def __getitem__(self, index):
        # Get image location
        img_loc = self.main_dir + self.all_imgs[index]
        # Represent image as a tensor
        img = np.load(img_loc)
        # scaling idea from https://www.mdpi.com/2076-3417/10/21/7837
        # set all air (<-1000) to 0 and all bone (>400) to 1, scale all other numbers to between [0,1] 
        img = np.where((self.HU_Lower <= img) & (img <= self.HU_Upper), (img - self.HU_Lower)/(self.HU_Upper - self.HU_Lower), img)
        img[img<self.HU_Lower] = 0
        img[img>self.HU_Upper] = 1
        img = self.transform(img) 
        return img
    
# Dirichlet VAE re-parameterisation trick: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9668955  
### Credit to Rachael Harkness (give proper citation) ###
class ResampleDir(nn.Module):
    def __init__(self, latent_dim, batch_size):
        global alpha_fill_value
        super(ResampleDir, self).__init__()
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.alpha_target = torch.full((batch_size, latent_dim), fill_value=alpha_fill_value, dtype=torch.float, device=device)#'cuda') #0.5
        
    def concentrations_from_logits(self, logits):
        alpha_c = torch.exp(logits)
        alpha_c = torch.clamp(alpha_c, min=1e-10, max=1e10)
        alpha_c = torch.log(1.+alpha_c)
        return alpha_c

    def dirichlet_kl_divergence(self, logits, eps=1e-10):
        alpha_c_pred = self.concentrations_from_logits(logits)

        alpha_0_target = torch.sum(self.alpha_target, axis=-1, keepdims=True)
        alpha_0_pred = torch.sum(alpha_c_pred, axis=-1, keepdims=True)

        term1 = torch.lgamma(alpha_0_target) - torch.lgamma(alpha_0_pred)
        term2 = torch.lgamma(alpha_c_pred + eps) - torch.lgamma(self.alpha_target + eps)

        term3_tmp = torch.digamma(self.alpha_target + eps) - torch.digamma(alpha_0_target + eps)
        term3 = (self.alpha_target - alpha_c_pred) * term3_tmp

        result = torch.squeeze(term1 + torch.sum(term2 + term3, keepdims=True, axis=-1))
        return result
 
    def prior_forward(self, logits): # analytical kld loss
        latent_vector = self.dirichlet_kl_divergence(logits)
        return latent_vector

    def sample(self, logits):
        alpha_pred = self.concentrations_from_logits(logits)   
        # rsample creates a differentiable sample
        dir_sample = torch.squeeze(Dirichlet(alpha_pred).rsample()) #1 # output to decoder 
        return dir_sample

   # def direct_kld_loss(self, alpha_pred):
   #     dir1 = Dirichlet(self.alpha_target)
   #     dir2 = Dirichlet(alpha)

   #     kld = dir2.log_prob(alpha_pred) - dir1.log_prob(alpha_pred)
   #     return kld
############################################################


# Concolutional Block
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Conv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Convolutional Transpose Block for upsampling (decoder)
class ConvTranspose(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTranspose, self).__init__()
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Convolutional Bilinear Upsampling Block    
#https://distill.pub/2016/deconv-checkerboard/
#  the checkerboard could be reduced by replacing transpose convolutions with bilinear upsampling
class ConvUpsampling(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvUpsampling, self).__init__()
        
        self.scale_factor = kernel_size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear')
        return self.conv(x)
    
# Main VAE Model
class DIR_VAE(nn.Module):
    def __init__(self):
        super(DIR_VAE, self).__init__()
        
        global base, latent_size
        # output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        self.encoder = nn.Sequential(
            Conv(1, base, 3, stride=1, padding=1),        # (64 - 3 + 2)/1 + 1  = 64
            Conv(base, 2*base, 3, stride=1, padding=1),   # 64
            Conv(2*base, 2*base, 3, stride=2, padding=1), # (64 - 3 + 2)/2 + 1 = 32
            Conv(2*base, 2*base, 3, stride=1, padding=1), # 32
            Conv(2*base, 4*base, 3, stride=2, padding=1), # (32 - 3 + 2)/2 + 1 = 16
            Conv(4*base, 4*base, 3, stride=1, padding=1), # 16
            Conv(4*base, 4*base, 3, stride=2, padding=1), # (16 - 3 + 2)/2 + 1 = 8
            nn.Conv2d(4*base, 32*base, 8),                # (8 - 8 + 0)/1 + 1 = 1
            nn.GELU(),
            nn.Flatten(),
            nn.Linear(in_features= 32*base, out_features=latent_size*base, bias=False),
            nn.BatchNorm1d(num_features=latent_size*base, momentum=0.9),
            nn.GELU()
        )
        
        self.alpha_fc = nn.Linear(in_features=latent_size*base, out_features=latent_size*base)
        
        # Conv2d_output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        # ConvTranspose_output_width =(input_width −1)*stride − 2*in_padding + dilation*(kernel_width−1) + out_padding +1
        self.decoder = nn.Sequential(
             nn.Linear(in_features=latent_size*base, out_features=32*base, bias=False),
             nn.BatchNorm1d(num_features=32*base, momentum=0.9),
             nn.GELU(),
             nn.Unflatten(1,(32*base,1,1)),
             nn.Conv2d(32*base, 32*base, 1),                       # (1 - 1)/1 + 1 = 1                              ## 32 64
             ConvTranspose(32*base, 4*base, 8),                    # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4         
             Conv(4*base, 4*base, 3, padding=1),                   # (8 - 3 + 2)/1 + 1 = 8                          ## 4 4 
             ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4         
             Conv(4*base, 2*base, 3, padding=1),                   # (16 - 3 + 2)/1 + 1 = 16                        ## 4 2 
             ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2         
             Conv(2*base, base, 3, padding=1),                     # 32                                             ## 2 1  
             ConvUpsampling(base, base, 4, stride=2, padding=1),    # (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64   ## 1 1 
             nn.Conv2d(base, 1, 3, padding=1),                     # 64                                             ## 1 1
             nn.Sigmoid() #nn.Tanh()
        )
    def encode(self, x):
        return self.encoder(x)    
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        x = self.encode(x)
        batch_size = x.shape[0]
        #print(x.shape[0])
        alpha = self.alpha_fc(x)
        resampler = ResampleDir(latent_size, batch_size)
        dirichlet_sample = resampler.sample(alpha) # This variable that follows a Dirichlet distribution
        recon_x = self.decoder(dirichlet_sample)   # can be interpreted as a probability that the sum is 1)
        return recon_x, alpha, dirichlet_sample
    

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, alpha, epoch):
        global annealing, beta, alpha_scalar, ssim_indicator, latent_size, ssim_scalar, base
        batch_size = x.shape[0]
        scale_factor = 1/(batch_size*base)
        # linear annealing: reduce the effect of KL divergence over time
        def linear_annealing(init, fin, step, annealing_steps):
            """Linear annealing of a parameter."""
            if annealing_steps == 0:
                return fin
            assert fin > init
            delta = fin - init
            annealed = min(init + delta * step / annealing_steps, fin)
            return annealed

        if annealing == 1:
            C = (linear_annealing(0, 1, epoch, epochs))
        if annealing == 0:
            C = 0
            
        # Calculating KL with Dirichlet prior and variational posterior distributions
        # Original paper:"Autoencodeing variational inference for topic model"-https://arxiv.org/pdf/1703.01488
        resampler = ResampleDir(latent_size*base, batch_size) 

        # 0.5 * sum(1 + log(logvar^2) - mu^2 - logvar^2)
        #kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * scale_factor
        kld = resampler.prior_forward(alpha)
        kld = torch.sum(kld) 
        
        l1_loss = nn.L1Loss(reduction='sum') 
        recon_loss = l1_loss(recon_x, x) * scale_factor
        #print("recon scale factor", scale_factor)

        if ssim_scalar == 2:
            ssim_scalar = batch_size

        if ssim_indicator == 0:
            recon_mix = recon_loss  

        # idea from https://arxiv.org/pdf/1511.08861.pdf     alpha*l1_loss + (1-aplha)*ssim_loss  *guassian kernal (not included!)
        if ssim_indicator == 1:
            # https://github.com/VainF/pytorch-msssim
            ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
            recon_mix = alpha_scalar*recon_loss + (1-alpha_scalar)*ssim_loss*ssim_scalar  

        if ssim_indicator == 2:
            # https://github.com/VainF/pytorch-msssim
            ssim_loss = 1 - ms_ssim(x, recon_x, data_range=1, win_size=3)
            recon_mix = alpha_scalar*recon_loss + (1-alpha_scalar)*ssim_loss*ssim_scalar  


        # idea from beta vae: https://openreview.net/pdf?id=Sy2fzU9gl
        beta_norm = (10*beta*latent_size)/(64*64*batch_size)
        #print("KLD scale factor", beta_norm)
        beta_vae_loss = recon_mix + beta_norm*(kld - C).abs()
        pure_loss = recon_loss + kld

        ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
        ms_ssim_score = ms_ssim(x, recon_x, data_range=1, win_size=3)
        if epoch%100==1 or (epoch < 3):
            print('recon loss: {:.4f}'.format(recon_loss.item()),'recon mix: {:.4f}'.format(recon_mix), 
                  'kld loss: {:.4f}'.format(kld.item()), 'kld loss scaled: {:.4f}'.format((kld - C).abs().item()),
                  'SSIM score: {:.4f}'.format(ssim_score.item()), 'MS-SSIM: {:.4f}'.format(ms_ssim_score.item()))
        return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss


def plot_results(results_path, filename):   #train_losses, test_losses
    '''
    method for plotting accuracy and loss graphs and saving file in given destination
    '''
    data = torch.load(os.path.join(results_path, "VAE_params.pt"))
    loss = data["train_losses"]
    val_loss = data["test_losses"]
    loss = loss[1:]
    val_loss = val_loss[1:]
    fig, ax1 = plt.subplots()
    plt.plot(loss, 'm', label = 'training loss')
    plt.plot(val_loss, 'g', label = 'test loss')
    plt.yscale("log")
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss')
    fig.savefig(os.path.join(results_path, filename))
    plt.show()
    plt.close()
    



def train(model, epoch, optimiser, sample_shape, train_loader):
    model.train()
    train_loss, beta_train_loss = 0, 0
    ssim_list = []
    for batch_idx, data in enumerate(train_loader):
        data = data.float().to(device)
        optimiser.zero_grad()
        batch_size = data.shape[0]
        recon_batch, alpha, dirichlet_sample = model(data)
        loss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch)
        ssim_list.append(ssim_score.item())
        loss.backward()
        train_loss += pure_loss.item()
        beta_train_loss += loss.item()
        optimiser.step()
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                pure_loss.item(), loss.item()))
        if math.isnan(loss):
            break

    if((epoch%50==1) or (epoch < 5) or (epoch==epochs-1)):
        print('12 Real Images')
        img_grid = make_grid(data[:12], nrow=4, padding=12, pad_value=-1)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu())
        plt.axis('off')
        plt.savefig(results_path + "/" + "visualise_real" + str(epoch) + '.png')
        plt.show()


        print('12 Reconstructed Images')
        img_grid = make_grid(recon_batch[:12], nrow=4, padding=12, pad_value=-1)
        plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu())
        plt.axis('off')
        plt.savefig(results_path + "/" + "visualise_reconstructed" + str(epoch) + '.png')
        plt.show()


        print('12 Synthetic Images')
        sample = torch.randn(sample_shape).to(device)
        sample.to(device)
        recon_rand_sample = model.decode(sample)
        img_grid = make_grid(recon_rand_sample[:12], nrow=4, padding=12, pad_value=-1)
        fig = plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu()) #, cmap='gray')
        plt.axis('off')
        plt.savefig(results_path + "/" + "visualise_synthetic" + str(epoch) + '.png') #, dpi=100)
        plt.show()
    
    train_loss /= len(train_loader.dataset)
    beta_train_loss /= len(train_loader.dataset)
    print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
    print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))

    return train_loss, ssim_mean, kld

def test(model, epoch, test_loader):
    global batch_size
    model.eval()
    test_loss, beta_test_loss = 0, 0
    ssim_list = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data.float().to(device)
            recon_batch, alpha, dirichlet_sample = model(data)
            testloss, recon_loss, kld, ssim_score, pure_loss = model.loss_function(recon_batch, data, alpha, epoch)
            test_loss += pure_loss.item()
            beta_test_loss += testloss.item()
            ssim_list.append(ssim_score.item())
            if math.isnan(testloss):
                break
            if (epoch%50 == 1) or (epoch == epochs - 1):
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    beta_test_loss /= len(test_loader.dataset)
    print('====> Pure Test Loss: {:.4f}'.format(test_loss))
    print('====> Beta Test Loss: {:.4f}'.format(beta_test_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Test SSIM: {:.4f}'.format(ssim_mean))
    return test_loss, ssim_mean

def early_stopping(counter, train_loss, test_loss, min_delta):
    if (test_loss - train_loss) > min_delta:
        counter += 1
        if counter % 5 == 0:
            print('Early Stopping Counter At:', counter)  
    return counter

def train_model(model, lr, epochs, sample_shape, Run, train_loader, test_loader):
    train_losses, test_losses, ssim_score_list = [], [], []
    optimiser = optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-4)
    # factor: how much to reduce learning rate, patients: how many epochs without improvement before reducing, threshold: for measuring the new optimum
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                           threshold=0.001, threshold_mode='abs')
    counter = 0
    for epoch in range(1, epochs + 1):
        train_loss, ssim_score, kld = train(model, epoch, optimiser, sample_shape, train_loader)
        test_loss, test_ssim = test(model, epoch, test_loader)
        scheduler.step(train_loss)
        counter = early_stopping(counter, train_loss, test_loss, min_delta=1)
        if counter > 25:
            print("At Epoch:", epoch)
            break
        if math.isnan(train_loss):
            print('Training stopped due to infinite loss')
            break
                
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        ssim_score_list.append(ssim_score)
    
    #np.save(results_path + '/' + 'train_ssim_score_{}'.format(Run), ssim_score_list)    
    #np.save(results_path + '/' + 'train_loss_{}'.format(Run), train_losses)    
    #np.save(results_path + '/' + 'test_loss_{}'.format(Run), test_losses)       
    torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "test_losses": test_losses}, save_results_path) 
        
    return test_loss, test_ssim

epochs = 400
        
# parameter_space = {"HU_UpperBound":[300, 400, 500, 600], "HU_LowerBound":[-1000, -800, -700, -500],
#                     "base":[32, 64, 128], "latent_size":[4, 8, 16, 32, 64], "annealing":[0, 1], "ssim_indicator":[0, 1, 2],
#                     "alpha":[0.15, 0.2, 0.3, 0.5, 0.7, 0.8, 0.85], "beta":[0.5, 0.8, 1, 1.5, 2, 5, 10, 20, 30, 50, 100, 250],
#                     "lr":[1e-7, 1e-6, 1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 0.001], 
#                     "batch_size":[64, 128, 256, 512], "ssim_scalar":[1,2], "alpha_fill_value":[0.6, 0.9, 0.99, 3]
#                   } 
parameter_space = {"HU_UpperBound":[400, 500, 600],      # Upper bound of Hounsfield units
                   "HU_LowerBound":[-1000, -800, -700], # Lower bound of Hounsfield units
                   "base":[18, 32],              # number of feature maps in convolutional layers
                   "latent_size": [4, 8, 16, 32], # base*latent_size = size of latent space
                   "annealing":[0, 1],                 # annealing on KL Divergence indicator (0 for NO or 1 for YES)
                   "ssim_indicator":[0, 1],          # SSIM indicator for loss function: 0 to not inlcude, 1 to include, 2 to use MS-SSIM
                   "batch_size":[64, 128, 256, 512],    # batch size (bigger for more stable training)
                   "alpha": [0.5, 0.7, 0.8],            # If using SSIM in loss function this is the balance between reconstruction loss (L1/MAE) and the other metric in alpha*L1_loss + (1-alpha)*other_metric
                   "beta":[0.8, 1, 2, 5, 10, 20, 30, 50], # multiplier for KL divergence, helps to disentangle latent space. Idea from beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl
                   "lr":[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 5e-3],    # learning rate (smaller number for slower training)
                   "ssim_scalar":[1, 2],                 # multiplier for SSIM: 1 for no upweighting, 2 for multiply by batch_size
                   "recon_scale_factor":[1, 2, 3],     # multiplier for reconstrction loss
                   "alpha_fill_value":[0.6, 0.9, 0.99, 3]
                   }


                  
ssim_list, loss_list = [], []

hyperparams_list = list(np.load(results_path + '/' + 'hyperparams_list.npy', allow_pickle=True))

#print('already tried', hyperparams_list)
print('number tried so far:', len(hyperparams_list))

hyperparams = {}
hyperparams["HU_UpperBound"] = random.choice(parameter_space["HU_UpperBound"])
hyperparams["HU_LowerBound"] = random.choice(parameter_space["HU_LowerBound"])
hyperparams["base"] = random.choice(parameter_space["base"])
hyperparams["latent_size"] = random.choice(parameter_space["latent_size"])
hyperparams["annealing"] = random.choice(parameter_space["annealing"])
hyperparams["ssim_indicator"] = random.choice(parameter_space["ssim_indicator"])
hyperparams["batch_size"] = random.choice(parameter_space["batch_size"])  
hyperparams["alpha"] = random.choice(parameter_space["alpha"])   
hyperparams["beta"] = random.choice(parameter_space["beta"])       
hyperparams["lr"] = random.choice(parameter_space["lr"])
hyperparams["ssim_scalar"] = random.choice(parameter_space["ssim_scalar"])
hyperparams["recon_scale_factor"] = random.choice(parameter_space["recon_scale_factor"])

if hyperparams in hyperparams_list:
    print("done before:", hyperparams)
    hyperparams["HU_UpperBound"] = random.choice(parameter_space["HU_UpperBound"])
    hyperparams["HU_LowerBound"] = random.choice(parameter_space["HU_LowerBound"])
    hyperparams["base"] = random.choice(parameter_space["base"])
    hyperparams["latent_size"] = random.choice(parameter_space["latent_size"])
    hyperparams["annealing"] = random.choice(parameter_space["annealing"])
    hyperparams["ssim_indicator"] = random.choice(parameter_space["ssim_indicator"])
    hyperparams["batch_size"] = random.choice(parameter_space["batch_size"])  
    hyperparams["alpha"] = random.choice(parameter_space["alpha"])   
    hyperparams["beta"] = random.choice(parameter_space["beta"])       
    hyperparams["lr"] = random.choice(parameter_space["lr"])
    hyperparams["ssim_scalar"] = random.choice(parameter_space["ssim_scalar"])
    hyperparams["recon_scale_factor"] = random.choice(parameter_space["recon_scale_factor"])

HU_UpperBound = hyperparams["HU_UpperBound"]
HU_LowerBound = hyperparams["HU_LowerBound"]
base = hyperparams["base"]
latent_size = hyperparams["latent_size"]
annealing = hyperparams["annealing"]      
ssim_indicator = hyperparams["ssim_indicator"]
batch_size = hyperparams["batch_size"]
alpha_scalar = hyperparams["alpha"] # had to renmae gamma dur to re-param technique
beta = hyperparams["beta"]
lr = hyperparams["lr"]
ssim_scalar = hyperparams["ssim_scalar"]  
alpha_fill_value = random.choice(parameter_space["alpha_fill_value"])
recon_scale_factor = hyperparams["recon_scale_factor"]

print("Using Hyperparams:", hyperparams) 
print("latent size:", latent_size*base)
print("alpha traget (fill value)", alpha_fill_value)

def vae_data_split(batch_size):
    global HU_UpperBound, HU_LowerBound, all_files_list
    def is_train(row,train,test):
        if row in train:
            return 'Train'
        else:
            return 'Test'
    meta = pd.read_csv(r"VAE_lung_lesion/Preprocessing/meta_mal_nonmal.csv")
    patient_id = list(np.unique(meta['patient_id']))
    train_patient , test_patient = train_test_split(patient_id,test_size = 0.3)
    meta['data_split']= meta['patient_id'].apply(lambda row : is_train(row,train_patient,test_patient))

    split = list(meta["data_split"])
    train_images, test_images = [], []
    for index, item in enumerate(split):
        if item == 'Train':
            train_images.append(all_files_list[index])
        if item == 'Test':
            test_images.append(all_files_list[index])
            
    print("Samples:     Train:", len(train_images), "   Test:", len(test_images))
    print("Proportions:       {: .3f},      {: .3f}".format(100*len(train_images)/13852, 100*len(test_images)/13852))
    train_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=train_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    test_images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=test_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
    train_loader = DataLoader(train_images, batch_size, shuffle=True)
    test_loader = DataLoader(test_images, batch_size, shuffle=False)
    return train_loader, test_loader

# Load image data using LoadImages classtrain_loader, test_loader = data_split(batch_size)

vae_model = DIR_VAE()
vae_model = vae_model.to(device)                     
print('parameter count:', parameter_count(DIR_VAE()))
train_loader, test_loader  = vae_data_split(batch_size)
test_loss, ssim_score = train_model(vae_model, lr=lr, epochs = epochs, sample_shape = (12, latent_size*base), Run=Run, train_loader=train_loader, test_loader=test_loader) 
plot_results(results_path, 'loss_graph_{}.jpg'.format(Run))
ssim_list.append(ssim_score)
loss_list.append(test_loss)
hyperparams_list.append(hyperparams)  
np.save(results_path + '/' + 'hyperparams_list', hyperparams_list)    

#idx = loss_list.index(min(loss_list))
print('Final Test Loss:', test_loss, 
      'Final SSIM:', ssim_score, 
      'Hyperparameters:', hyperparams)         

vae_test_loss = test_loss

images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=all_files_list, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound)
image_loader = DataLoader(images, batch_size, shuffle=False)
vae_model.eval()
MSE = nn.MSELoss(reduction='mean')
l1_loss = nn.L1Loss(reduction='mean') 
mus, log_vars, reconstructions = [], [], []
SSIM_list, MSE_list, L1_list = [], [], []
if not math.isnan(vae_test_loss):
    with torch.no_grad():
        for batch_idx, data in enumerate(image_loader):
            data = data.float().to(device)
            reconstructions_batch, alpha, dirichlet_sample = vae_model(data)
            # save latent vectors
            for mu in alpha:
                mus.append(mu.tolist()) #torch.squeeze(torch.squeeze(alpha, dim=1), dim=1).tolist())  
            # calculate SSIM
            SSIM_batch = ssim(data, reconstructions_batch, data_range=1, nonnegative_ssim=True)
            SSIM_list.append(np.array(SSIM_batch.cpu()).item())
            # calculate MSE
            MSE_batch = MSE(data, reconstructions_batch)
            MSE_list.append(np.array(MSE_batch.cpu()).item())
            # calculate MAE
            L1_batch = l1_loss(data, reconstructions_batch)
            L1_list.append(np.array(L1_batch.cpu()).item())

        
    print('Number of latent vectors', len(mus))
    print('Mean Squared Error', np.mean(MSE_list))
    print('Mean Absolute Error', np.mean(L1_list))
    print('Mean SSIM', np.mean(SSIM_list))

    np.save(results_path + '/' + 'latent_vectors_{}'.format(Run), mus) 
        
    metrics_list = [ssim_score, test_loss, np.mean(MSE_list), np.mean(L1_list), hyperparams]        
        
        
        
# MLP Classifier 
class LoadData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        super(LoadData, self).__init__()
        # store the raw tensors
        self._x = x
        self._y = y

    def __len__(self):
        # a dataset must know it size
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index, :]
        y = self._y[index, :]
        return x, y

def data_split(n, meta, latent_vectors, labels, batch_size):
    def which_set(row,data_split):
        for i, dataset in enumerate(data_split):
            if row in dataset:
                return i
    random.seed(42)
    patient_id = list(np.unique(meta['patient_id']))
    data_split, used = [], []
    for i in range(n):
        temp_set = []
        while len(temp_set) < len(patient_id)//n:
            index = random.choice(patient_id)
            if index not in used:
                used.append(index)
                temp_set.append(index)
        if i == n-1:
            for pat_id in patient_id:
                if pat_id not in used:
                    temp_set.append(pat_id)    
        data_split.append(temp_set)
        
    
    meta['data_split'] = meta['patient_id'].apply(lambda row : which_set(row,data_split))
    print(len(latent_vectors), len(labels))
    split = list(meta["data_split"])
    cross_val_data, cross_val_labels = [], []
    for i in range(n):
        vecs, labs = [], []
        for index, item in enumerate(split):
            if item == i:
                vecs.append(torch.tensor(latent_vectors[index]))
                labs.append(torch.tensor(labels[index]))            
        vecs = torch.stack(vecs)
        labs = torch.unsqueeze(torch.stack(labs), 1)       
        cross_val_data.append(vecs)
        cross_val_labels.append(labs)
    
    return cross_val_data, cross_val_labels

def Cross_Validation(run, n, meta, latent_vectors, labels, batch_size):
    def other_index(exclude, n):
        index = []
        for i in range(n):
            if i not in exclude:
                index.append(i)
        return index
    def find_subsets(run, n):
        if run != n-1:
            return other_index([n-2-run, n-1-run], n), n-2-run, n-1-run
        if run == n-1:
            return other_index([0, run], n), run, 0

    def concat_train_data(indices, datasets):
        train_data = []
        for idx in indices:
            train_data.append(datasets[idx])
        return train_data
    
    loss_list, accuracy_list, results_list, auc_list = [], [], [], []
    cross_val_data, cross_val_labels = data_split(n, meta, latent_vectors, labels, batch_size)    
    
    train_data, train_labels = [], []
    cross_val_split = find_subsets(run, n)
    for i in cross_val_split[0]:
        train_data.append(cross_val_data[i])
        train_labels.append(cross_val_labels[i])
    train_data = torch.cat(train_data,dim=0)
    train_labels = torch.cat(train_labels,dim=0)

    train_dataset = LoadData(train_data, train_labels)
    val_index = cross_val_split[1]
    test_index = cross_val_split[2]
    validation_dataset = LoadData(cross_val_data[val_index], cross_val_labels[val_index])
    test_dataset = LoadData(cross_val_data[test_index], cross_val_labels[test_index])

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False) 
    
    return train_loader, validation_loader, test_loader


if not math.isnan(vae_test_loss):
    latent_vectors = np.load(r"/nobackup/mm17b2k/Dirichlet_VAE_RandomSearch/latent_vectors_{}.npy".format(Run))
    ambiguous = np.load(r"VAE_lung_lesion/MLP/latent_vectors/ambiguous.npy")
    latent_vectors2 = []
    for i, vec in enumerate(latent_vectors):
        if i not in ambiguous:
            latent_vectors2.append(torch.tensor(vec))
    latent_vectors2 = torch.stack(latent_vectors2)
    #np.save(r"VAE_lung_lesion/VAE_RandomSearch/latent_vectors2_{}.npy".format(Run), latent_vectors2)

def get_predictions(predictions, threshold):
    preds = []
    for pred in predictions:
        if pred >= threshold:
            preds.append([1])  # p closer to 1
        if pred < threshold:
            preds.append([0]) # p close to 0
    return torch.Tensor(preds)

np.set_printoptions(linewidth=110)

def confusion_matrix(outputs, train_labels, threshold):
    '''
    method of plotting confusions matrix with first numerical counts and second percentage values
    '''
    ndata = len(outputs)
    labels = np.squeeze(train_labels)
    labels = np.array([int(lab) for lab in labels])
    # convert outputs to numpy array 
    if type(outputs) == torch.Tensor:
        preds = np.array(outputs.detach())
    else:
        preds = outputs
    
    predictions = []
    for pred in preds:
        if pred >= threshold:
            predictions.append(1)  # p closer to 1
        if pred < threshold:
            predictions.append(0) # p close to 0
    predictions = np.array(predictions)   
    nclasses = 2
    # one-hot encoding all predictions (1 in position of prediction and rest 0s)
    # confusion matrix (cm) 30x30 of all 0s
    cm = np.zeros((nclasses, nclasses)) # cm with counts
    cm2 = np.zeros((nclasses, nclasses)) # cm with percentage
    # loop through matrix and replace values with number of outputs in each category
    for i in range(nclasses):
        for j in range(nclasses):
            cm[i, j] = np.sum(np.where(predictions == i, 1, 0) * np.where(labels == j, 1, 0))   # 1 x 1 if label and prediction match
            cm2 = cm/cm.sum(0) # normalise so columns (predictions of each class) sum to 1

  #  print("The confusion matrix is:")
  #  print(cm)
  #  print(cm2)
  #  print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)
    return cm

def stats(loader, model, threshold):
    '''
    function to calculate validation accuracy and loss
    '''
    correct = 0
    total = 0
    running_loss = 0
    n = 1    # counter for number of minibatches
    output_list = []
    label_list = []
    loss_fn = nn.BCELoss()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.float().to(device)
            labels = labels.float().to(device)
            model.eval()
            outputs = model(images)    

            # accumulate loss
            running_loss += loss_fn(outputs, labels)
            n += 1

            # accumulate data for accuracy
            #_, predicted = torch.max(outputs.data, 1)
            predicted = get_predictions(outputs.data, threshold)
            predicted = predicted.to(device)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            output_list.append(outputs.cpu())
            label_list.append(labels.cpu())
        output_list = np.concatenate(output_list)
        label_list = np.concatenate(label_list)
    return running_loss.cpu()/n, correct/total, output_list, label_list

def evaluation_metrics(tp,fp,tn,fn):
    if tp == 0 and fp == 0:
        fp = 1
    if tp == 0 and fn == 0:
        fn = 1
    if tn == 0 and fp == 0:
        fp = 1
    precision = tp/(tp+fp)
    print('Precision,', 'proprotion of malignant predictions that are true:', precision,)
    recall = tp/(tp+fn)
    print('Recall,', 'proportion of tumours identified:', recall)
    specificity = tn/(tn+fp)
    print('Specificity,', 'proportion of non-cancerous lesions identified:', specificity)
    f1 = 2*((precision*recall)/(precision+recall))
    print('F1 score:', f1)
    
    results = [precision, recall, specificity, f1]
    return results

def average_metrics(results_list):
    precision, recall, specificity, f1 = [], [], [], []
    for result in results_list:
        precision.append(result[0])
        recall.append(result[1])
        specificity.append(result[2])
        f1.append(result[3])
    print('Average Precision: {}, Recall: {}, Specificity: {}, and F1 Score: {}'.format(np.mean(precision), np.mean(recall),
                                                                                        np.mean(specificity), np.mean(f1)))
    average_results = [np.mean(precision), np.mean(recall), np.mean(specificity), np.mean(f1)]
    return average_results

def plot_results(results_path, filename, epoch):
    '''
    method for plotting accuracy and loss graphs and saving file in given destination
    '''
    data = torch.load(results_path)
    statsrec = data["stats"]
    fig, ax1 = plt.subplots()
    plt.plot(statsrec[0][:epoch], 'm', label = 'training loss', )
    plt.plot(statsrec[2][:epoch], 'g', label = 'validation loss' )
    plt.legend(loc='lower right')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Training and validation loss, and validation accuracy')
    ax2=ax1.twinx()
    ax2.plot(statsrec[1][:epoch], 'b', label = 'training accuracy')
    ax2.plot(statsrec[3][:epoch], 'r', label = 'validation accuracy')
    ax2.set_ylabel('accuracy')
    plt.legend(loc='upper right')
    #fig.savefig(filename)
    plt.show()
    plt.close()

def early_stopping(counter, train_loss, validation_loss, min_delta):
    if (validation_loss - train_loss) > min_delta:
        counter += 1
        if counter % 10 == 0 or counter == 25:
            print('early stopping counter at:', counter)
    return counter

def train_model(nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams):
    global latent_size, base
    threshold = mlp_hyperparams["threshold"]
    lr = mlp_hyperparams["lr"]
    size1 = mlp_hyperparams["layer_sizes"][0]
    size2 = mlp_hyperparams["layer_sizes"][1]
    size3 = mlp_hyperparams["layer_sizes"][2]
    dropout = mlp_hyperparams["dropout"]
    Depth = mlp_hyperparams["Depth"]
    
    if Depth == 4:
        model = nn.Sequential(
            nn.Linear(latent_size*base,size1),
            nn.GELU(),
            nn.BatchNorm1d(size1, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size1,size2),
            nn.GELU(),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size2,size3),
            nn.GELU(),
            nn.BatchNorm1d(size3, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size3,1),
            nn.Sigmoid()
        )

    if Depth == 5:
        model = nn.Sequential(
            nn.Linear(latent_size*base,size1),
            nn.GELU(),
            nn.BatchNorm1d(size1, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size1,size2),
            nn.GELU(),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),


            nn.Linear(size2,size2),
            nn.GELU(),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size2,size3),
            nn.GELU(),
            nn.BatchNorm1d(size3, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),

            nn.Linear(size3,1),
            nn.Sigmoid()
        )
    
    model.to(device)
    save_results_path = r"VAE_lung_lesion/VAE_RandomSearch/MLP.pt"
    statsrec = np.zeros((4,nepochs))

    loss_fn = nn.BCELoss()
    optimiser = optimizer = optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                               threshold=0.001, threshold_mode='abs')
    counter = 0
    for epoch in range(1,nepochs+1):  # loop over the dataset multiple times
        correct = 0          # number of examples predicted correctly (for accuracy)
        total = 0            # number of examples
        running_loss = 0.0   # accumulated loss (for mean loss)
        n = 0                # number of minibatches
        model.train()
        for data in train_loader:
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            if inputs.shape[0] == 1:
                continue   
            # Zero the parameter gradients
            optimiser.zero_grad()

            # Forward
            outputs = model(inputs)
            # Backward, and update parameters
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # accumulate data for accuracy
            predicted = get_predictions(outputs.data, threshold)
            predicted = predicted.to(device)
            total += labels.size(0)    # add in the number of labels in this minibatch
            correct += (predicted == labels).sum().item()  # add in the number of correct labels
            
            # accumulate loss
            running_loss += loss.item()
            n += 1

        # collect together statistics for this epoch
        ltrn = running_loss/n
        atrn = correct/total 
        results = stats(valid_loader, model, threshold)
        lval, aval = results[0], results[1]
        val_outputs, val_labels = results[2], results[3]
        statsrec[:,epoch-1] = (ltrn, atrn, lval.item(), aval)
        if epoch % 75 == 0 or epoch == 1 or epoch == nepochs - 1 or counter == 24:
            print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  validation loss: {lval: .3f} validation accuracy: {aval: .1%}")
            
        test_results = stats(test_loader,model, threshold)
        test_loss, test_acc = test_results[0], test_results[1]
        test_outputs, test_labels = test_results[2], test_results[3]
        if epoch % 75 == 0 or epoch == nepochs - 1 or counter == 25:
            print('test loss:', test_loss.item(),'test accuracy:', test_acc*100, '%')  
        scheduler.step(test_loss)
        counter = early_stopping(counter, ltrn, lval, min_delta=0.25)
        if counter > 25:
            print("At Epoch:", epoch)
            break
    # save network parameters, losses and accuracy
    torch.save({"stats": statsrec}, save_results_path) #"state_dict": model.state_dict()
    plot_results(save_results_path, "fulltrainset.jpg", epoch)
    test_cm = confusion_matrix(test_outputs, test_labels, threshold)
    auc = metrics.roc_auc_score(test_labels, test_outputs)
    print('AUC is:', auc)
    results = evaluation_metrics(test_cm[1,1], test_cm[0,1], test_cm[0,0], test_cm[1,0])
    
    return test_loss.item(), test_acc, results, auc

break_indicator = 0

def test_hyperparams(parameter_space, data, vae_test_loss):
    global latent_vectors2, Run, break_indicator
    accuracy_list, loss_list, auc_list, mlp_hyperparams_list, results_list = [], [], [], [], []
    mlp_runs = 0
    num_tried = 0
    if not math.isnan(vae_test_loss):
        if data == 1:
            # malignant vs non-malignant   
            meta = pd.read_csv(r"VAE_lung_lesion/Preprocessing/meta_mal_nonmal.csv")
            latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(Run), allow_pickle=True)
            labels = np.load(r"VAE_lung_lesion/MLP/latent_vectors/labels2.npy")
        if data == 2:
           # malignant vs benign 
            meta = pd.read_csv(r"VAE_lung_lesion/Preprocessing/meta_mal_ben.csv")
            latent_vectors = latent_vectors2 #np.load(r"VAE_lung_lesion/MLP/latent_vectors/mu2original.npy")
            labels = np.load(r"VAE_lung_lesion/MLP/latent_vectors/labels3.npy")   
               
        while len(mlp_hyperparams_list) < 25:
            mlp_runs +=1
            print("Attempt:", mlp_runs)
            print("Number of completed attempts:", num_tried)
            if mlp_runs == 100:
                break_indicator = 1
                break
            random.seed(time.time())
            mlp_hyperparams = {}
            mlp_hyperparams["threshold"] = random.choice(parameter_space["threshold"])
            mlp_hyperparams["lr"] = random.choice(parameter_space["lr"])
            mlp_hyperparams["layer_sizes"] = random.choice(parameter_space["layer_sizes"])
            mlp_hyperparams["dropout"] = random.choice(parameter_space["dropout"])
            batch_size = random.choice(parameter_space["batch_size"])
            mlp_hyperparams["batch_size"] = batch_size
            mlp_hyperparams["Depth"] = random.choice(parameter_space["Depth"])  

            if mlp_hyperparams in mlp_hyperparams_list:
                    continue

            print(mlp_hyperparams)


            train_start = perf_counter()
            CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
            for run in range(5):  # 5-fold cross validation
                train_loader, valid_loader, test_loader = Cross_Validation(run, 5, meta, latent_vectors, labels, batch_size)
                loss, accuracy, results, auc = train_model(350, train_loader, valid_loader, test_loader, mlp_hyperparams)
                if accuracy < 0.6:
                    print("########## This is not a good candidate for cross validation ##########")
                    break 
                CV_loss.append(loss)
                CV_accuracy.append(accuracy)
                CV_results.append(results)
                CV_auc.append(auc)

          
            train_stop= perf_counter()
            print('training time', train_stop - train_start)  
            
            if len(CV_auc) == 5  or mlp_runs < 3:
                print(".................Cross Validation Averages.................")
                print("AUC:", np.mean(CV_auc), "Loss:", np.mean(CV_loss), "Accuracy:", np.mean(CV_accuracy), "Results:", average_metrics(CV_results))
                auc_list.append(np.mean(CV_auc))
                loss_list.append(np.mean(CV_loss))
                accuracy_list.append(np.mean(CV_accuracy))
                mlp_hyperparams_list.append(mlp_hyperparams)
                results_list.append(average_metrics(CV_results))
                if len(CV_auc) == 5:
                    num_tried +=1
                    print("Number of completed attempts:", num_tried)
                        
                auc_list = np.nan_to_num(auc_list).tolist()
                idx = auc_list.index(max(auc_list))
                print("Best so far", "Based on AUC:",
                'AUC:', max(auc_list), 
                'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
                'Performance Metrics:', results_list[idx],'Index:', idx, 
                'Hyperparameters:', mlp_hyperparams_list[auc_list.index(max(auc_list))])
                
                accuracy_list = np.nan_to_num(accuracy_list).tolist()
                idx2 = accuracy_list.index(max(accuracy_list))
                print("Best so far", "Based on Accuracy",
                'AUC:', auc_list[idx2], 'Test Loss:', loss_list[idx2], 'Test Accuracy:', accuracy_list[idx2], 
                'Performance Metrics:', results_list[idx2],'Index:', idx2, 
                'Hyperparameters:', mlp_hyperparams_list[accuracy_list.index(max(accuracy_list))])
        

        if len(auc_list) > 1:   
            auc_list = np.nan_to_num(auc_list).tolist()
            idx = auc_list.index(max(auc_list))
            print("Based on AUC:",
              'AUC:', max(auc_list), 
              'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
              'Performance Metrics:', results_list[idx],
              'Index:', idx, 
              'Hyperparameters:', mlp_hyperparams_list[auc_list.index(max(auc_list))])
        
            accuracy_list = np.nan_to_num(accuracy_list).tolist()
            idx2 = accuracy_list.index(max(accuracy_list))
            print("Based on Accuracy",
              'AUC:', auc_list[idx2], 'Test Loss:', loss_list[idx2], 'Test Accuracy:', accuracy_list[idx2], 
              'Performance Metrics:', results_list[idx2],'Index:', idx2, 
              'Hyperparameters:', mlp_hyperparams_list[accuracy_list.index(max(accuracy_list))])
        
            if data == 1:
                Mal_NonMal = [auc_list[idx],loss_list[idx],accuracy_list[idx],results_list[idx], mlp_hyperparams_list[idx]]
                Mal_Ben = [0,0,0,0,0]
                
                Mal_NonMal2 = [auc_list[idx2],loss_list[idx2],accuracy_list[idx2],results_list[idx2], mlp_hyperparams_list[idx2]]
                Mal_Ben2 = [0,0,0,0,0]
                
            if data == 2:
                Mal_NonMal = [0,0,0,0,0]
                Mal_Ben = [auc_list[idx],loss_list[idx],accuracy_list[idx],results_list[idx], mlp_hyperparams_list[idx]]  
                
                Mal_NonMal2 = [0,0,0,0,0]
                Mal_Ben2 = [auc_list[idx2],loss_list[idx2],accuracy_list[idx2],results_list[idx2], mlp_hyperparams_list[idx2]]  

    if math.isnan(vae_test_loss) or break_indicator == 1:
        Mal_Ben = [0,0,0,0,0]
        Mal_NonMal = [0,0,0,0,0]
        Mal_Ben2 = [0,0,0,0,0]
        Mal_NonMal2 = [0,0,0,0,0]

    return Mal_NonMal, Mal_Ben, Mal_NonMal2, Mal_Ben2 #auc_list, loss_list, accuracy_list, hyperparams_list,


# parameter_space = {"threshold":[0.6,0.55,0.5,0.45,0.4], "lr":[1e-7,1e-6,1e-5], 
#                    "layer_sizes":[[8192, 4096, 1024], [4096, 4096, 2048], [4096, 4096, 1024], [4096, 2048, 1024], [4096, 1024, 1024],
#                                   [2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
#                                   [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
#                                   [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
#                                   [512, 256, 256]],
#                    "dropout":[0.05,0.1,0.3,0.4,0.5,0.55], "batch_size":[32,64,128,256,512],
#                    "Depth":[4,5]
#                   }
parameter_space = {"threshold":[0.6,0.55,0.5,0.45,0.4], "lr":[1e-6,1e-5,1e-4],
                   "layer_sizes":[[2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
                                  [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
                                  [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
                                  [512, 256, 256]],
                   "dropout":[0.2,0.4,0.5,0.6], "batch_size":[32,64,128,256,512],
                   "Depth":[4, 5]
                   }


Mal_NonMal, a, Mal_NonMal2, a2 = test_hyperparams(parameter_space, data=1, vae_test_loss=vae_test_loss)

b, Mal_Ben, b2, Mal_Ben2 = test_hyperparams(parameter_space, data=2, vae_test_loss=vae_test_loss)

print('Mean Squared Error (across full dataset):', np.mean(MSE_list))
print('Mean SSIM', np.mean(SSIM_list))
print('Mean Absolute Error', np.mean(L1_list))

print("malignant vs non-malignant AUC",
      'AUC:', Mal_NonMal[0], 'Test Loss:', Mal_NonMal[1], 'Test Accuracy:', Mal_NonMal[2], 
      'Performance Metrics:', Mal_NonMal[3], "Hyperparams", Mal_NonMal[4]) 

print("malignant vs non-malignant Accuracy",
      'AUC:', Mal_NonMal2[0], 'Test Loss:', Mal_NonMal2[1], 'Test Accuracy:', Mal_NonMal2[2], 
      'Performance Metrics:', Mal_NonMal2[3], "Hyperparams", Mal_NonMal2[4]) 

print("malignant vs benign AUC",
      'AUC:', Mal_Ben[0], 'Test Loss:', Mal_Ben[1], 'Test Accuracy:', Mal_Ben[2], 
      'Performance Metrics:', Mal_Ben[3], "Hyperparams", Mal_Ben[4]) 

print("malignant vs benign Accuracy",
      'AUC:', Mal_Ben2[0], 'Test Loss:', Mal_Ben2[1], 'Test Accuracy:', Mal_Ben2[2], 
      'Performance Metrics:', Mal_Ben2[3], "Hyperparams", Mal_Ben2[4]) 


#print(vae_test_loss)
if not math.isnan(vae_test_loss):
    full_results_list = list(np.load(results_path + '/' + 'Candidate_Dirichlet_VAE_and_Classifier_Results.npy', allow_pickle=True))
  #  print("[SSIM, Test Loss, MSE, VAE hyperparameters],",
  #        "[AUC, Test Loss, Test Accuracy, Performance Metrics: precision, recall, specificity, f1, MLP Hyperparameters]", full_results_list)
    np.save(results_path + '/' + 'run.npy', [Run])
    results = [metrics_list, Mal_NonMal, Mal_Ben]
    full_results_list.append(results)
    np.save(results_path + '/' + 'Candidate_Dirichlet_VAE_and_Classifier_Results.npy', full_results_list) 
    print("this run:", "[SSIM, Test Loss, MSE, VAE hyperparameters]",
          "[AUC, Test Loss, Test Accuracy, Performance Metrics: precision, recall, specificity, f1, MLP Hyperparameters]")
    print(results)
    print("alpha fill value", alpha_fill_value)
