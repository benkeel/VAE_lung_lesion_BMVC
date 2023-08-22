'''
This file is the script used to train the VAE on ARC (High Performance Computer)
Make a VAE train loaded which only contains Mal/Ben for the retrainining
'''

IMAGE_DIR = r"VAE_lung_lesion/Images"
results_path = r"/nobackup/mm17b2k/VAE_RandomSearch"
save_results_path = r"/nobackup/mm17b2k/VAE_RandomSearch/VAE_params.pt"
meta_location = r"VAE_lung_lesion/Preprocessing"
labels_path = r"VAE_lung_lesion/MLP/latent_vectors"
mlp_save_results_path = r"/nobackup/mm17b2k/VAE_RandomSearch/MLP.pt"

# torch for data loading and for model building
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

#  First save 'Run' after first run then un hashtag to keep track.
Run = np.load(results_path + '/' + 'run.npy', allow_pickle=True)
Run = Run[0]
Run+=1
print("Run:", Run)

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

# Compute number of parameters in VAE model
def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Create class to load image data from files
class LoadImages(Dataset):
    def __init__(self, main_dir, files_list, HU_Upper, HU_Lower, labels):
        # Set the loading directory
        self.main_dir = main_dir

        # Transforms
        self.transform = transforms.Compose([transforms.ToTensor()])

        # Get list of all image file names
        self.all_imgs = files_list

        # Get HU limits
        self.HU_Upper = HU_Upper
        self.HU_Lower = HU_Lower

        # Get labels
        self.labels = labels


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
        label = self.labels[index]
        return img, label


# Definition of VAE model

# Class for convolutional block (for encoder):
# 2d convolution followed by 2d batch normalisation and GELU activation
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

# Class for convolutional transpose block (for decoder):
# 2d convolutional transpose followed by 2d batch normalisation and GELU activation
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

# Class for convolutional upsampling block (for decoder):
# 2d convolutional transpose followed by 2d batch normalisation and GELU activation
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

# Class for final convolutional transpose block: changes activation from GELU to Sigmoid. Errors are not symmetric so using a symmetrical actvation function helps improve model performance. Tanh will also likely work and can easily change below to test this.
class ConvTransposeFinal(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvTransposeFinal, self).__init__()

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.Sigmoid(), # nn.Tanh()
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)

# Main VAE model code (using conv blocks from above)
class VAE(nn.Module):
    def __init__(self, hyperparams):
        super(VAE, self).__init__()
        base = hyperparams['base']
        latent_size = hyperparams['latent_size']
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
            nn.GELU()
        )
        self.encoder_mu = nn.Conv2d(32*base, latent_size*base, 1) # (1 - 1)/1 + 1 = 1    ## 32*base = 32*4 = 128  ### 1*512 = 512
        self.encoder_logvar = nn.Conv2d(32*base, latent_size*base, 1)

        # Conv2d_output_width = [ (input_width - kernel_width + 2*padding) / stride ] + 1
        # ConvTranspose_output_width =(input_width −1)*stride − 2*in_padding + dilation*(kernel_width−1) + out_padding +1
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_size*base, 32*base, 1),                       # (1 - 1)/1 + 1 = 1                              ## 32 64
            ConvTranspose(32*base, 4*base, 8),                    # (1-1)*1 + 2*0 + 1(8-1) + 0 + 1  = 8     ## 64 4
            Conv(4*base, 4*base, 3, padding=1),                   # (8 - 3 + 2)/1 + 1 = 8                          ## 4 4
            ConvUpsampling(4*base, 4*base, 4, stride=2, padding=1),# (8-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 16     ## 4 4
            Conv(4*base, 2*base, 3, padding=1),                   # (16 - 3 + 2)/1 + 1 = 16                        ## 4 2
            ConvUpsampling(2*base, 2*base, 4, stride=2, padding=1),# (16-1)*2 - 2*1 + 1(4-1) + 0 + 1 = 32    ## 2 2
            Conv(2*base, base, 3, padding=1),                     # 32                                             ## 2 1
            ConvUpsampling(base, base, 4, stride=2, padding=1),    # (32-1)*2 - 2*1 + 1*(4-1) + 0 + 1 = 64   ## 1 1
            nn.Conv2d(base, 1, 3, padding=1),                    # 64                                             ## 1
            nn.Sigmoid() # nn.Tanh()
        )

    # Function to encode an input and return latent vectors
    def encode(self, x):
        x = self.encoder(x)
        return self.encoder_mu(x), self.encoder_logvar(x)

    # Reparameterisation function which takes mean and log variance latent vectors and produces an input to the decoder
    # Reparameterisation trick:
    # mean + (sample from N(0,1))*standard_deviation
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std) # mean=0 , std=1
        #z = mu + logvar*self.N.sample(mu.shape)
        return mu + eps*std

    # Function to decode/reconstruct a given reparameterised latent vector
    def decode(self, z):
        return self.decoder(z)

    # Forward pass of VAE network, given input return the reconstruction and both corresponding latent vectors
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar



def plot_results(results_path, filename, vae_results_path):   #train_losses, test_losses
    '''
    method for plotting accuracy and loss graphs and saving file in given destination
    Input:
    - results_path: where to save the error plot
    - filename: name given to loss graph file
    '''
    data = torch.load(vae_results_path)
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

def linear_annealing(init, fin, step, annealing_steps):
    '''
    Function for linear annealing of a parameter. Used here for KL divergence.
    Input:
    init: value which is subtracted on first step (default 0)
    fin: final value subtracted on last step (default 1)
    step: current epoch
    annealing_steps: epoch to stop annealing
    '''
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed

def kl_annealing(epoch, max_epochs, weight_max):
    # Compute the annealing weight using an exponential decay function
    weight = 1 + weight_max * (1 - np.exp(-5 * epoch / max_epochs))
    return weight

# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar, epoch, hyperparams, retrain_indicator, labels):
    '''
    Function to calcuate loss between a batch of input data and VAE reconstructions

    Input:
    - recon_x: reconstructed data
    - x: input data
    - mu: mean latent vector
    - logvar: log variance latent vector
    - epoch: current epoch id (used for annealing)
    - hyperparams: dictionary of hyperparameters of model (several influence loss function)
    - retrain_indicator: if 0 then on just train VAE, if 1 then jointly train VAE with MLP classifier

    Output:
    - beta_vae_loss: loss which has been combined with other metrics/scaled
    - recon_loss: L1 loss (mean absolute error)
    - kld: KL divergence
    - ssim_score: SSIM score (structural similarity)
    - pure_loss: un-scaled/altered loss
    '''
    global mlp_model
    # get hyperparameters
    base = hyperparams["base"]
    annealing = hyperparams["annealing"]
    ssim_indicator = hyperparams["ssim_indicator"]
    batch_size = hyperparams["batch_size"]
    alpha = hyperparams["alpha"]
    beta = hyperparams["beta"]
    ssim_scalar = hyperparams["ssim_scalar"]
    recon_scale_factor = hyperparams["recon_scale_factor"]


    # To help generalise weighting across different number of feature maps and batch sizes
    scale_factor = 1/(batch_size*base)

    # linear annealing: reduce the effect of KL divergence over time
    if annealing == 1:
        annealing_weight = kl_annealing(epoch, epochs, 1)
        C = (linear_annealing(0, 1, epoch, epochs))
    if annealing == 0:
        C = 0
        annealing_weight = 1

    # KL Divergence = -0.5 * sum(1 + log(variance) - mu^2 - exponential(logvar))
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) * scale_factor
    kld_mean = -0.5 * torch.mean(torch.sum(1 + logvar - mu**2 - torch.exp(logvar), dim=1))

    # sum may achieve a bigger range in quaity of reconstructions
    l1_loss = nn.L1Loss(reduction='sum')
    # reduction: mean - may get a most consistent quality across the dataset
    l1_loss_mean= nn.L1Loss(reduction='mean')

    recon_loss_mean = l1_loss_mean(recon_x, x)
    recon_loss = l1_loss(recon_x, x) * scale_factor * recon_scale_factor

    # If ssim_scaler = 1 then do not scale SSIM
    # If ssim_scaler = x then scale SSIM by x


    # If ssim_scaler = 2 then upweight SSIM by s.f. batch_size
    if ssim_scalar == 2:
        ssim_scalar = batch_size


    # Choose whether to include SSIM / (other metric if replaced for time series metric)
    # ssim_indicator = 0 if not including
    if ssim_indicator == 0:
        recon_mix = recon_loss

    # ssim_indicator = 1 if including, and weight by alpha / (1-alpha)
    if ssim_indicator == 1:
        ssim_loss = 1 - ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
        #print("Recon scaled", alpha*recon_loss, "SSIM scaled", (1-alpha)*ssim_loss*ssim_scalar)
        recon_mix = alpha*recon_loss + (1-alpha)*ssim_loss*ssim_scalar

    # Use multiple-scale SSIM (MS-SSIM) instead of standard SSIM https://github.com/VainF/pytorch-msssim
    if ssim_indicator == 2:
        ssim_loss = 1 - ms_ssim(x, recon_x, data_range=1, win_size=3)
        recon_mix = alpha*recon_loss + (1-alpha)*ssim_loss*ssim_scalar


    # idea from beta vae: https://openreview.net/pdf?id=Sy2fzU9gl
    beta_norm = (beta*latent_size*base)/(64*64)


    beta_vae_loss = recon_mix + beta_norm*(kld - C).abs()   #kld*annealing_weight
    pure_loss = recon_loss_mean + kld_mean

    ssim_score = ssim(x, recon_x, data_range=1, nonnegative_ssim=True)
    ms_ssim_score = ms_ssim(x, recon_x, data_range=1, win_size=3)
    if epoch == 1:
        #print("Recon:", recon_loss.item(), "Recon Mean:", recon_loss_mean.item(), "SSIM Loss/Score:", ssim_loss.item(), ssim_score.item(), "SSIM Loss Scaled:", ssim_loss.item()*ssim_scalar, "KLD:", kld.item(), "KLD Mean/Un-scaled", kld_mean.item(), kld.item()*(batch_size*base), "Beta norm", beta_norm)
        print("SSIM Loss:", ssim_loss.item(), "SSIM Loss Scaled:", ssim_loss.item()*ssim_scalar,
              "Recon Mean/Un-scaled:", recon_loss_mean.item(), recon_loss.item()/(scale_factor * recon_scale_factor),
              "KLD Mean/Un-scaled/Annealed", kld_mean.item(), kld.item()*(batch_size*base), kld.item()*annealing_weight,
              "Beta norm", beta_norm)
    if retrain_indicator == 1:
        loss_fn = nn.BCELoss(reduction="mean")
        mu_mlp = torch.squeeze(torch.squeeze(mu, dim=2),dim=2)
        y = mlp_model(mu_mlp)
        y = torch.squeeze(y, dim=1)
        mlp_loss = loss_fn(y, labels)
        beta_vae_loss = beta_vae_loss + mlp_loss

    if retrain_indicator == 0:
        mlp_loss = 0
    if epoch%100==1 or (epoch < 3):
        print('recon loss: {:.4f}'.format(recon_loss.item()),'recon mix: {:.4f}'.format(recon_mix.item()),
              'kld loss: {:.4f}'.format(kld.item()),
              'SSIM score: {:.4f}'.format(ssim_score.item()), 'MS-SSIM: {:.4f}'.format(ms_ssim_score.item()),
              'pure loss: kld + recon =: {:.4f} + {:.4f} = {:.4f}'.format(kld_mean.item(), recon_loss_mean.item(), pure_loss.item()), 'MLP loss: {:.4f}'.format(mlp_loss))

    return beta_vae_loss, recon_loss, kld, ssim_score, pure_loss, mlp_loss




def train(epoch, optimiser, sample_shape, train_loader, vae_model, hyperparams, retrain_indicator):
    '''
    Function to run one epoch of training on given data

    Input:
    - epoch: the epoch id
    - optimiser: optimiser used (default Adam)
    - train_loader: DataLoader for training data
    - vae_model: VAE model used
    - hyperparams: dictionary of hyperparameters of model
    - retrain_indicator: if 0 then on just train VAE, if 1 then jointly train VAE with MLP classifier

    Output:
    - Current training loss
    - SSIM score
    - KL divergence
    '''
    print('Current Epoch:', epoch)
    # Put model in training mode
    vae_model.train()
    # Create variables to store metrics
    train_loss, beta_train_loss, mlp_loss = 0, 0, 0
    ssim_list = []
    # Iterate through training DataLoader and apply model and loss function
    for batch_idx, dataset in enumerate(train_loader):
        data, labels  = dataset
        data = data.float().to(device)
        labels = labels.float().to(device)
        optimiser.zero_grad()  # Initailise the optimisation function
        recon_batch, mu, logvar = vae_model(data) # Feed one batch through VAE model

        # Calculate loss and save various metrics
        loss, recon_loss, kld, ssim_score, pure_loss, mlp_loss = loss_function(recon_batch, data, mu, logvar, epoch, hyperparams, retrain_indicator, labels)
        ssim_list.append(ssim_score.item())
        loss.backward() # Backpropagate loss
        # Update running losses
        train_loss += pure_loss.item()
        beta_train_loss += loss.item()
        mlp_loss += mlp_loss
        # Update optimisation function
        optimiser.step()
        # Print current state of model (metrics)
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tPure Loss: {:.6f}, Beta Loss: {:.6f}, MLP Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                pure_loss.item(), loss.item(), mlp_loss))
        if math.isnan(loss):
            break

    if((epoch%50==1) or (epoch < 20) or (epoch==epochs-1)):
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
        recon_rand_sample = vae_model.decode(sample)
        img_grid = make_grid(recon_rand_sample[:12], nrow=4, padding=12, pad_value=-1)
        fig = plt.figure(figsize=(10,5))
        plt.imshow(img_grid[0].detach().cpu()) #, cmap='gray')
        plt.axis('off')
        plt.savefig(results_path + "/" + "visualise_synthetic" + str(epoch) + '.png') #, dpi=100)
        plt.show()

    # Give epoch summary of metrics using running losses
    train_loss /= len(train_loader.dataset)
    beta_train_loss /= len(train_loader.dataset)
    mlp_loss /= len(train_loader.dataset)
    print('====> Epoch {}: Average Train Loss: {:.4f}'.format(epoch, train_loss))
    print('====> Average Beta Train Loss: {:.4f}'.format(beta_train_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Train SSIM: {:.4f}'.format(ssim_mean))
    print('====> Average Train MLP Loss: {:.4f}'.format(mlp_loss))
    return train_loss, ssim_mean, kld

def test(epoch, test_loader, vae_model, hyperparams, retrain_indicator):
    '''
    Function to test model after each epoch

    Input:
    - epoch: the epoch id
    - test_loader: DataLoader for test data
    - vae_model: VAE model used
    - hyperparams: dictionary of hyperparameters of model
    - retrain_indicator: if 0 then on just train VAE, if 1 then jointly train VAE with MLP classifier

    Output:
    - Test loss
    - SSIM score
    '''
    global mlp_model, threshold, epochs
    # Put model in testing mode
    vae_model.eval()
    # Create variables to store metrics
    test_loss, beta_test_loss, mlp_loss = 0, 0, 0
    ssim_list = []

    correct, total, mlp_running_loss = 0, 0, 0
    n = 1    # counter for number of minibatches
    mlp_output_list = []
    mlp_label_list = []
    mlp_loss_fn = nn.BCELoss()

    # Iterate through test DataLoader and apply model and loss function
    with torch.no_grad():
        for i, dataset in enumerate(test_loader):
            data, labels = dataset
            data = data.float().to(device)
            labels = labels.float().to(device)
            # Feed one batch through VAE model
            recon_batch, mu, logvar = vae_model(data)
            # Calculate loss and save various metrics
            testloss, recon_loss, kld, ssim_score, pure_loss, mlp_loss = loss_function(recon_batch, data, mu, logvar, epoch, hyperparams, retrain_indicator, labels)
            # Update running losses
            test_loss += pure_loss.item()
            beta_test_loss += testloss.item()
            ssim_list.append(ssim_score.item())
            mlp_loss += mlp_loss
            # if test loss is NaN then stop
            if math.isnan(testloss):
                break
            if (epoch%50 == 1) or (epoch == epochs - 1):
                if i == 0:
                    n = min(data.size(0), 8)
                    comparison = torch.cat([data[:n], recon_batch.view(batch_size, 1, 64, 64)[:n]])
                    save_image(comparison.cpu(),
                               results_path + '/' + str(epoch) + '.png', nrow=n)
            if retrain_indicator == 1:
                mlp_model.eval()
                mu_mlp = torch.squeeze(torch.squeeze(mu, dim=2),dim=2)
                mlp_outputs = mlp_model(mu_mlp)
                mlp_outputs = torch.squeeze(mlp_outputs, dim=1)
                mlp_labels = labels
                #mlp_labels = torch.squeeze(labels,dim=1)
                # accumulate loss
                mlp_running_loss += mlp_loss_fn(mlp_outputs, mlp_labels)
                n += 1
                # accumulate data for accuracy
                predicted = get_predictions(mlp_outputs.data, threshold)
                predicted = predicted.to(device)
                predicted = torch.squeeze(predicted, dim=1)
                total += labels.size(0)    # add in the number of labels in this minibatch
                correct += (predicted == labels).sum().item()  # add in the number of correct labels
                mlp_output_list.append(mlp_outputs.cpu())
                mlp_label_list.append(labels.cpu())


    if retrain_indicator == 1:
        test_mlp_outputs = np.concatenate(mlp_output_list)
        test_mlp_labels = np.concatenate(mlp_label_list)
        mlp_model.train()
        test_loss_mlp, test_acc_mlp = mlp_running_loss.cpu()/n, correct/total
        test_cm = confusion_matrix(test_mlp_outputs, test_mlp_labels, threshold)
        auc = metrics.roc_auc_score(test_mlp_labels, test_mlp_outputs)
        mlp_results = evaluation_metrics(test_cm[1,1], test_cm[0,1], test_cm[0,0], test_cm[1,0])
        if epoch % 2 == 0 or epoch == epochs - 1 or epoch < 25:
            print('AUC:', auc,
                  'Test Loss:', test_loss_mlp.item(), 'Test Accuracy:', test_acc_mlp, '%',
                  'Performance Metrics:', 'Precision:', mlp_results[0], "Recall:", mlp_results[1], 'Specificity:', mlp_results[2], "F1:", mlp_results[3])

    # Give epoch summary of metrics using running losses
    test_loss /= len(test_loader.dataset)
    beta_test_loss /= len(test_loader.dataset)
    mlp_loss /= len(test_loader.dataset)
    print('====> Pure Test Loss: {:.4f}'.format(test_loss))
    print('====> Beta Test Loss: {:.4f}'.format(beta_test_loss))
    ssim_mean = np.mean(ssim_list)
    print('====> Average Test SSIM: {:.4f}'.format(ssim_mean))
    print('====> Average Train MLP Loss: {:.4f}'.format(mlp_loss))

    return test_loss, ssim_mean


def early_stopping(counter, train_loss, test_loss, min_delta):
    '''
    Function to stop the training if difference between training and testing loss becomes too big
    Input:
    - counter: current epochs where gap between train/test loss is too big
    - train_loss and test_loss
    - min_delta: size of gap permitted

    Return:
    - counter: after adding 1 if permitted gap between losses is exceeded.
    '''
    if (test_loss - train_loss) > min_delta:
        counter += 1
        if counter % 5 == 0:
            print('Early Stopping Counter At:', counter)
    return counter

def train_VAE_model(model, epochs, sample_shape, Run, train_loader, test_loader, hyperparams, retrain_indicator):
    '''
    Function to run training procedure of multiple epochs
    Input:
    - model: VAE model
    - epochs: number of epochs to train over
    - Run: current run number in hyperparameter training regime
    - train_loader and test_loader: DataLoaders for train/test data
    - hyperparams: dictionary of hyperparameters of model
    - retrain_indicator: if 0 then on just train VAE, if 1 then jointly train VAE with MLP classifier

    Output:
    - test_loss: to check whether loss has exploaded (infinite/NaN) as will need to stop training
    - test_ssim: SSIM for quick summary of how well the model has done
    '''
    lr = hyperparams["lr"]
    # Lower the learning rate when re-training with VAE with MLP for fine tuning
    lr_copy = lr
    if retrain_indicator == 1:
        lr = lr/2
    if retrain_indicator == 0:
        lr = lr_copy
    train_losses, test_losses, ssim_score_list = [], [], []
    optimiser = optim.Adam(model.parameters(), lr=lr) #, weight_decay=1e-4)
    # factor: how much to reduce learning rate, patience: how many epochs without improvement before reducing, threshold: for measuring the new optimum
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20,
                                                           threshold=0.001, threshold_mode='abs')          # threshold: 0.01?
    counter = 0
    # Loop through epochs and run train and test functions
    for epoch in range(1, epochs + 1):
        train_loss, ssim_score, kld = train(epoch, optimiser, sample_shape, train_loader, model, hyperparams, retrain_indicator)
        test_loss, test_ssim = test(epoch, test_loader, model, hyperparams, retrain_indicator)
        # update learning rate scheduler based on training loss
        scheduler.step(train_loss)
        # update counter for early stopping: if reaches 20 steps where difference between train and test loss is greater than 1 then halt training
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
    if retrain_indicator == 0:
        torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "test_losses": test_losses}, save_results_path)

    if retrain_indicator == 1:
        retrained_results_path = r"/nobackup/mm17b2k/VAE_RandomSearch/retrained_VAE_params.pt"
        torch.save({"state_dict": model.state_dict(), "train_losses": train_losses, "test_losses": test_losses}, retrained_results_path)

    return test_loss, test_ssim



# Hyperparameter set up

# parameter_space = {"HU_UpperBound":[400, 500, 600],      # Upper bound of Hounsfield units
#                     "HU_LowerBound":[-1000, -800, -700], # Lower bound of Hounsfield units
#                     "base":[32, 64],              # number of feature maps in convolutional layers
#                     "latent_size": [8, 16, 32], # base*latent_size = size of latent space
#                     "annealing":[1],                 # annealing on KL Divergence indicator (0 for NO or 1 for YES)
#                     "ssim_indicator":[1, 2],          # SSIM indicator for loss function: 0 to not inlcude, 1 to include, 2 to use MS-SSIM
#                     "batch_size":[64, 128, 256, 512],    # batch size (bigger for more stable training)
#                     "alpha":[0.2, 0.3, 0.5, 0.7, 0.8],            # If using SSIM in loss function this is the balance between reconstruction loss (L1/MAE) and the other metric in alpha*L1_loss + (1-alpha)*other_metric
#                     "beta":[1, 2, 5, 10, 20, 30, 50], # multiplier for KL divergence, helps to disentangle latent space. Idea from beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl
#                     "lr":[1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 5e-3],    # learning rate (smaller number for slower training)
#                     "ssim_scalar":[1, 2],                 # multiplier for SSIM: 1 for no upweighting, 2 for multiply by batch_size
#                     "recon_scale_factor":[1, 2, 3]     # multiplier for reconstrction loss
#                     }

# Set them to be deterministic for now
parameter_space = {"HU_UpperBound":[400],      # Upper bound of Hounsfield units
                   "HU_LowerBound":[-500], # Lower bound of Hounsfield units
                   "base":[128],   #[24, 32, 64, 128],             # number of feature maps in convolutional layers
                   "latent_size": [4], # base*latent_size = size of latent space
                   "annealing":[0],                 # annealing on KL Divergence indicator (0 for NO or 1 for YES)
                   "ssim_indicator":[1],          # SSIM indicator for loss function: 0 to not inlcude, 1 to include, 2 to use MS-SSIM
                   "batch_size":[64],    # batch size (bigger for more stable training)
                   "alpha":[0.5],            # If using SSIM in loss function this is the balance between reconstruction loss (L1/MAE) and the other metric in alpha*L1_loss + (1-alpha)*other_metric
                   "beta":[1.5], # multiplier for KL divergence, helps to disentangle latent space. Idea from beta-VAE: https://openreview.net/pdf?id=Sy2fzU9gl
                   "lr":[1e-5],    # learning rate (smaller number for slower training)
                   "ssim_scalar":[2],                 # multiplier for SSIM: 1 for no upweighting, 2 for multiply by batch_size
                   "recon_scale_factor":[1]     # multiplier for reconstrction loss
                   }


ssim_list, loss_list = [], []

hyperparams_list = list(np.load(results_path + '/' + 'hyperparams_list.npy', allow_pickle=True))
#metrics_list = list(np.load(results_path + '/' + 'VAEMetrics.npy', allow_pickle=True))
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
alpha = hyperparams["alpha"]
beta = hyperparams["beta"]
lr = hyperparams["lr"]
ssim_scalar = hyperparams["ssim_scalar"]
recon_scale_factor = hyperparams["recon_scale_factor"]


print("Using Hyperparams:", hyperparams)
print("latent size:", latent_size*base)
vae_hyperparams = hyperparams

def vae_data_split(batch_size, hyperparams, all_files_list, meta_location):
    HU_LowerBound = hyperparams["HU_LowerBound"]
    HU_UpperBound = hyperparams["HU_UpperBound"]

    def is_train(row,train,test):
        if row in train:
            return 'Train'
        else:
            return 'Test'
    meta = pd.read_csv(meta_location + '//' + 'meta_mal_nonmal.csv')
    patient_id = list(np.unique(meta['patient_id']))
    train_patient, test_patient = train_test_split(patient_id,test_size = 0.3)
    meta['data_split']= meta['patient_id'].apply(lambda row : is_train(row,train_patient, test_patient))


    split = list(meta["data_split"])
    labels = np.load(labels_path + '//' + 'labels2.npy')
    train_images, train_labels, test_images, test_labels = [], [], [], []

    for index, item in enumerate(split):
        if item == 'Train':
            train_images.append(all_files_list[index])
            train_labels.append(labels[index])
        if item == 'Test':
            test_images.append(all_files_list[index])
            test_labels.append(labels[index])

    print("Samples:     Train:", len(train_images), "   Test:", len(test_images))
    print("Proportions:       {: .3f},      {: .3f}".format(100*len(train_images)/13852, 100*len(test_images)/13852))
    train_dataset = LoadImages(main_dir=IMAGE_DIR + '/', files_list=train_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound, labels=train_labels)
    test_dataset = LoadImages(main_dir=IMAGE_DIR + '/', files_list=test_images, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound, labels=test_labels)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    return train_loader, test_loader

# Load image data using LoadImages class train_loader, test_loader = data_split(batch_size)
epochs = 400
vae_model = VAE(hyperparams)
vae_model = vae_model.to(device)
print('parameter count:', parameter_count(VAE(hyperparams)))
train_start = perf_counter()
vae_train_loader, vae_test_loader  = vae_data_split(batch_size, hyperparams, all_files_list, meta_location)
test_loss, ssim_score = train_VAE_model(vae_model, epochs = epochs, sample_shape = (12, latent_size*base, 1, 1), Run=Run, train_loader=vae_train_loader, test_loader=vae_test_loader, hyperparams=hyperparams, retrain_indicator=0)
train_stop= perf_counter()
VAE_training_time = train_stop - train_start
print('VAE Training Time', VAE_training_time)
plot_results(results_path, 'loss_graph_{}.jpg'.format(Run), save_results_path)
ssim_list.append(ssim_score)
loss_list.append(test_loss)
hyperparams_list.append(hyperparams)
np.save(results_path + '/' + 'hyperparams_list', hyperparams_list)

#idx = loss_list.index(min(loss_list))
print('Final Test Loss:', test_loss,
      'Final SSIM:', ssim_score,
      'Hyperparameters:', hyperparams)

vae_test_loss = test_loss
labels = np.load(labels_path + '//' + 'labels2.npy')

def evaluate_VAE(vae_model, labels):
    images = LoadImages(main_dir=IMAGE_DIR + '/', files_list=all_files_list, HU_Upper=HU_UpperBound, HU_Lower=HU_LowerBound, labels=labels)
    image_loader = DataLoader(images, batch_size, shuffle=False)
    vae_model.eval()
    MSE = nn.MSELoss(reduction='mean')
    l1_loss = nn.L1Loss(reduction='mean')
    mus, log_vars, reconstructions = [], [], []
    SSIM_list, MSE_list, L1_list = [], [], []
    if not math.isnan(vae_test_loss):
        with torch.no_grad():
            for batch_idx, dataset in enumerate(image_loader):
                data, labels = dataset
                data = data.float().to(device)
                labels = labels.float().to(device)
                reconstructions_batch, mu_batch, log_var_batch = vae_model(data)
                # save latent vectors
                # change this from append to .extend? using tensors and change for metrics after
                for mu in mu_batch:
                    mus.append(torch.squeeze(torch.squeeze(mu, dim=1), dim=1).tolist())
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

        metrics_list = [np.mean(SSIM_list), test_loss, np.mean(MSE_list), np.mean(L1_list), hyperparams]
        #np.save(results_path + '/' + 'VAEMetrics', metrics_list)
        return metrics_list

VAE1_metrics = evaluate_VAE(vae_model, labels)


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

def get_mal_ben_latent_vecs():
    if not math.isnan(vae_test_loss):
        latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(Run), allow_pickle=True)
        ambiguous = np.load(labels_path + '//' + "ambiguous.npy")
        latent_vectors2 = []
        for i, vec in enumerate(latent_vectors):
            if i not in ambiguous:
                latent_vectors2.append(torch.Tensor(vec))
        latent_vectors2 = torch.stack(latent_vectors2)
        #np.save(r"VAE_lung_lesion/VAE_RandomSearch/latent_vectors2_{}.npy".format(run), latent_vectors2)
        return latent_vectors2
latent_vectors2 = get_mal_ben_latent_vecs()

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
    #   print('Precision,', 'proprotion of malignant predictions that are true:', precision,)
    recall = tp/(tp+fn)
    #   print('Recall,', 'proportion of tumours identified:', recall)
    specificity = tn/(tn+fp)
    #   print('Specificity,', 'proportion of non-cancerous lesions identified:', specificity)
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

def plot_MLP_results(results_path, filename, epoch):
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
    #plt.show()
    plt.close()

def early_stopping(counter, train_loss, validation_loss, min_delta):
    if (validation_loss - train_loss) > min_delta:
        counter += 1
        if counter % 10 == 0 or counter == 25:
            print('early stopping counter at:', counter)
    return counter

def train_MLP_model(nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams):
    global latent_size, base, mlp_save_results_path
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
    statsrec = np.zeros((4,nepochs))

    loss_fn = nn.BCELoss()
    optimiser = optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)  # ? should this be here
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, threshold=0.001, threshold_mode='abs')
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
        if epoch % 5 == 0 or epoch == 1 or epoch == nepochs - 1 or counter == 25:
            print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  validation loss: {lval: .3f} validation accuracy: {aval: .1%}")

        test_results = stats(test_loader, model, threshold)
        test_loss, test_acc = test_results[0], test_results[1]
        test_outputs, test_labels = test_results[2], test_results[3]
        if epoch % 25 == 0 or epoch == nepochs - 1 or counter == 25:
            print('test loss:', test_loss.item(),'test accuracy:', test_acc*100, '%')
        scheduler.step(ltrn)  # reduce learning rate based on epoch average training loss
        counter = early_stopping(counter, ltrn, lval, min_delta=0.4)
        if counter > 25:
            print("At Epoch:", epoch)
            break
    # save network parameters, losses and accuracy
    torch.save({"stats": statsrec, "state_dict": model.state_dict()}, mlp_save_results_path)
    plot_MLP_results(mlp_save_results_path, "fulltrainset.jpg", epoch)
    test_cm = confusion_matrix(test_outputs, test_labels, threshold)
    auc = metrics.roc_auc_score(test_labels, test_outputs)
    print('AUC is:', auc)
    results = evaluation_metrics(test_cm[1,1], test_cm[0,1], test_cm[0,0], test_cm[1,0])

    return test_loss.item(), test_acc, results, auc

break_indicator = 0

def test_hyperparams(parameter_space, data, vae_test_loss, meta_location, labels_path, num_runs, nepochs):
    global latent_vectors2, Run, break_indicator, batch_size
    accuracy_list, loss_list, auc_list, mlp_hyperparams_list, results_list = [], [], [], [], []
    mlp_runs = 0
    num_tried = 0
    if not math.isnan(vae_test_loss):
        if data == 1:
            # malignant vs non-malignant
            meta = pd.read_csv(meta_location + '//' + 'meta_mal_nonmal.csv')
            latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(Run), allow_pickle=True)
            print("Using Latent Vecs:", latent_vectors[0][:20])
            labels = np.load(labels_path + '//' + 'labels2.npy')
        if data == 2:
            # malignant vs benign
            meta = pd.read_csv(meta_location + '//' + 'meta_mal_ben.csv')
            latent_vectors = latent_vectors2 #np.load(r"VAE_lung_lesion/MLP/latent_vectors/mu2original.npy")
            labels = np.load(labels_path + '//' + 'labels3.npy')

        while len(mlp_hyperparams_list) < num_runs:
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
            #batch_size = random.choice(parameter_space["batch_size"])
            #mlp_hyperparams["batch_size"] = batch_size
            mlp_hyperparams["Depth"] = random.choice(parameter_space["Depth"])

            if mlp_hyperparams in mlp_hyperparams_list:
                continue

            print(mlp_hyperparams)


            train_start = perf_counter()
            CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
            for run in range(5):  # 5-fold cross validation
                train_loader, valid_loader, test_loader = Cross_Validation(run, 5, meta, latent_vectors, labels, batch_size)
                loss, accuracy, results, auc = train_MLP_model(nepochs, train_loader, valid_loader, test_loader, mlp_hyperparams) # 350 epochs
                if accuracy < 0.65:
                    print("########## This is not a good candidate for cross validation ##########")
                    break
                CV_loss.append(loss)
                CV_accuracy.append(accuracy)
                CV_results.append(results)
                CV_auc.append(auc)
            train_stop= perf_counter()
            print('training time', train_stop - train_start)

            if len(CV_auc) == 5 or mlp_runs < 3:
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
            accuracy_list= np.nan_to_num(accuracy_list).tolist()
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
#                    "dropout":[0.4,0.5,0.55,0.6], "batch_size":[32,64,128,256,512],
#                    "Depth":[4,5]
#                    }
parameter_space = {"threshold":[0.6,0.55,0.5,0.45,0.4], "lr":[1e-6,1e-5],
                   "layer_sizes":[[2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
                                  [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
                                  [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
                                  [512, 256, 256]],
                   "dropout":[0.2,0.4,0.5,0.6],
                   "Depth":[4, 5]
                   }

train_start = perf_counter()
Mal_NonMal, a, Mal_NonMal2, a2 = test_hyperparams(parameter_space, data=1, vae_test_loss=vae_test_loss, meta_location=meta_location, labels_path=labels_path, num_runs=15, nepochs=150)
train_stop= perf_counter()
MLP1_training_time = train_stop - train_start
print("Malignant vs Non-Malignant MLP Training Time:", MLP1_training_time)

# train_start = perf_counter()
# b, Mal_Ben, b2, Mal_Ben2 = test_hyperparams(parameter_space, data=2, vae_test_loss=vae_test_loss, meta_location=meta_location, labels_path=labels_path, num_runs=12, nepochs=150)
# train_stop= perf_counter()
# MLP2_training_time = train_stop - train_start
# print("Malignant vs Benign MLP Training Time:", MLP2_training_time)
#Mal_Ben = [0,0,0,0,0]
#Mal_Ben2 = [0,0,0,0,0]


print('Mean Squared Error (across full dataset):', VAE1_metrics[2])
print('Mean SSIM', VAE1_metrics[0])
print('Mean Absolute Error', VAE1_metrics[3])

print("malignant vs non-malignant AUC",
      'AUC:', Mal_NonMal[0], 'Test Loss:', Mal_NonMal[1], 'Test Accuracy:', Mal_NonMal[2],
      'Performance Metrics:', Mal_NonMal[3], "Hyperparams", Mal_NonMal[4])

print("malignant vs non-malignant Accuracy",
      'AUC:', Mal_NonMal2[0], 'Test Loss:', Mal_NonMal2[1], 'Test Accuracy:', Mal_NonMal2[2],
      'Performance Metrics:', Mal_NonMal2[3], "Hyperparams", Mal_NonMal2[4])

# print("malignant vs benign AUC",
#       'AUC:', Mal_Ben[0], 'Test Loss:', Mal_Ben[1], 'Test Accuracy:', Mal_Ben[2],
#       'Performance Metrics:', Mal_Ben[3], "Hyperparams", Mal_Ben[4])

# print("malignant vs benign Accuracy",
#       'AUC:', Mal_Ben2[0], 'Test Loss:', Mal_Ben2[1], 'Test Accuracy:', Mal_Ben2[2],
#       'Performance Metrics:', Mal_Ben2[3], "Hyperparams", Mal_Ben2[4])



def MLP_matching_VAE_data_split(vae_model, vae_train_loader, vae_test_loader, batch_size):
    global vae_test_loss
    # Get same train/test sets as VAE
    vae_model.eval()
    mu_train, labels_train, mu_test, labels_test = [], [], [], []
    if not math.isnan(vae_test_loss):
        with torch.no_grad():
            for batch_idx, dataset in enumerate(vae_train_loader):
                data, labels = dataset
                data = data.float().to(device)
                labels = labels.float().to(device)
                reconstructions_batch, mu_batch, log_var_batch = vae_model(data)
                # save latent vectors
                # change this from append to .extend? using tensors and change for metrics after
                for mu in mu_batch:
                    mu_train.append(torch.squeeze(torch.squeeze(mu, dim=1), dim=1))
                for label in labels:
                    labels_train.append(label)
            for batch_idx, dataset in enumerate(vae_test_loader):
                data, labels = dataset
                data = data.float().to(device)
                labels = labels.float().to(device)
                reconstructions_batch, mu_batch, log_var_batch = vae_model(data)
                # save latent vectors
                # change this from append to .extend? using tensors and change for metrics after
                for mu in mu_batch:
                    mu_test.append(torch.squeeze(torch.squeeze(mu, dim=1), dim=1))
                for label in labels:
                    labels_test.append(label)

        train_data = torch.stack(mu_train)
        train_labels = torch.unsqueeze(torch.stack(labels_train), 1)
        train_dataset = LoadData(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)


        test_data = torch.stack(mu_test)
        test_labels = torch.unsqueeze(torch.stack(labels_test), 1)
        dataset = LoadData(test_data, test_labels)
        test_size = int(len(dataset) * 0.8)
        valid_size = len(dataset) - test_size
        # split the dataset into test and validation datasets
        test_dataset, valid_dataset = random_split(dataset, [test_size, valid_size], generator=torch.Generator().manual_seed(42))
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

def retrain_VAE_with_MLP(vae_hyperparams, mlp_hyperparams, data, mlp_retrain_train_loader, mlp_retrain_valid_loader, mlp_retrain_test_loader):
    global latent_size, base, latent_vectors2, mlp_save_results_path, vae_train_loader, vae_test_loader, vae_model, mlp_model, batch_size, epochs
    if data == 1:
        # malignant vs non-malignant
        meta = pd.read_csv(meta_location + '//' + 'meta_mal_nonmal.csv')
        latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(Run), allow_pickle=True)
        labels = np.load(labels_path + '//' + 'labels2.npy')
    if data == 2:
        # malignant vs benign
        meta = pd.read_csv(meta_location + '//' + 'meta_mal_ben.csv')
        latent_vectors = latent_vectors2
        labels = np.load(labels_path + '//' + 'labels3.npy')

    threshold = mlp_hyperparams["threshold"]
    lr = mlp_hyperparams["lr"]
    size1 = mlp_hyperparams["layer_sizes"][0]
    size2 = mlp_hyperparams["layer_sizes"][1]
    size3 = mlp_hyperparams["layer_sizes"][2]
    dropout = mlp_hyperparams["dropout"]
    Depth = mlp_hyperparams["Depth"]
    
    if Depth == 4:
        mlp_model = nn.Sequential(
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
        mlp_model = nn.Sequential(
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

    mlp_model.to(device)
    loss, accuracy, results, auc = train_MLP_model(200, mlp_retrain_train_loader, mlp_retrain_valid_loader, mlp_retrain_test_loader, mlp_hyperparams)
    mlp_model.load_state_dict(torch.load(mlp_save_results_path, map_location=torch.device('cpu'))["state_dict"])
    print('MLP parameter count:', parameter_count(mlp_model))
    mlp_model.train()
    #mlp_model.eval()  #Q: should MLP be in eval mode
    pre_trained_vae_results_path = save_results_path
    retrained_results_path = r"/nobackup/mm17b2k/VAE_RandomSearch/retrained_VAE_params.pt"
    vae_model = VAE(vae_hyperparams)
    print('VAE parameter count:', parameter_count(VAE(hyperparams)))
    vae_model = vae_model.to(device)
    vae_model.load_state_dict(torch.load(pre_trained_vae_results_path, map_location=torch.device('cpu'))["state_dict"])
    vae_model.train()
    epochs = 150
    test_loss, ssim_score = train_VAE_model(model = vae_model, epochs = epochs, sample_shape = (12, latent_size*base, 1, 1), Run=Run, train_loader=vae_train_loader, test_loader=vae_test_loader, hyperparams=vae_hyperparams, retrain_indicator=1)

    plot_results(results_path, 'loss_graph_part2_{}.jpg'.format(Run), retrained_results_path)
    

best_MLP_hyperparams_mal_nonmal = Mal_NonMal[4]

print("Using best MLP hyperparams: ", best_MLP_hyperparams_mal_nonmal)
threshold = best_MLP_hyperparams_mal_nonmal["threshold"]

train_start= perf_counter()
mlp_retrain_train_loader, mlp_retrain_valid_loader, mlp_retrain_test_loader = MLP_matching_VAE_data_split(vae_model, vae_train_loader, vae_test_loader, batch_size)

retrain_VAE_with_MLP(vae_hyperparams, best_MLP_hyperparams_mal_nonmal, data=1, mlp_retrain_train_loader=mlp_retrain_train_loader, mlp_retrain_valid_loader=mlp_retrain_valid_loader, mlp_retrain_test_loader=mlp_retrain_test_loader)

train_stop= perf_counter()
VAE_MLP_training_time = train_stop - train_start
print('M v ¬M VAE with Classifier Loss Training Time', VAE_MLP_training_time)

VAE2_metrics = evaluate_VAE(vae_model, labels)

train_start = perf_counter()
best_MLP_hyperparams_mal_nonmal_copy = {}
for key, value in best_MLP_hyperparams_mal_nonmal.items():
    best_MLP_hyperparams_mal_nonmal_copy[key] = [value]

Mal_NonMal_retrained, a, Mal_NonMal2_retrained, a2 = test_hyperparams(best_MLP_hyperparams_mal_nonmal_copy, data=1, vae_test_loss=vae_test_loss, meta_location=meta_location, labels_path=labels_path, num_runs=1, nepochs=250) #20
train_stop= perf_counter()
MLP3_training_time = train_stop - train_start
print("Malignant vs Non-Malignant Re-trained MLP Training Time:", MLP3_training_time)

train_start = perf_counter()
Mal_NonMal_retrained, a, Mal_NonMal2_retrained, a2 = test_hyperparams(parameter_space, data=1, vae_test_loss=vae_test_loss, meta_location=meta_location, labels_path=labels_path, num_runs=12, nepochs=100)
train_stop= perf_counter()
MLP4_training_time = train_stop - train_start
print("2nd Search: Malignant vs Non-Malignant Re-trained MLP Training Time:", MLP4_training_time)



#best_MLP_hyperparams_mal_ben = Mal_Ben[4]

#print("Using best MLP hyperparams: ", best_MLP_hyperparams_mal_ben)
#threshold = best_MLP_hyperparams_mal_ben["threshold"]

#train_start= perf_counter()
#retrain_VAE_with_MLP(vae_hyperparams, best_MLP_hyperparams_mal_ben, data=2, mlp_retrain_train_loader=mlp_retrain_train_loader, mlp_retrain_valid_loader=mlp_retrain_valid_loader, mlp_retrain_test_loader=mlp_retrain_test_loader)
#train_stop= perf_counter()
#VAE_MLP_training_time2 = train_stop - train_start
#print('M v B VAE with Classifier Loss Training Time', VAE_MLP_training_time2)

#VAE3_metrics = evaluate_VAE(vae_model, labels)

# train_start = perf_counter()
# best_MLP_hyperparams_mal_ben_copy = {}
# for key, value in best_MLP_hyperparams_mal_ben.items():
#     best_MLP_hyperparams_mal_ben_copy[key] = [value]

# #best_MLP_hyperparams_mal_ben = Mal_Ben[4]
# Mal_Ben_retrained, a, Mal_Ben2_retrained, a2 = test_hyperparams(best_MLP_hyperparams_mal_ben_copy, data=2, vae_test_loss=vae_test_loss, meta_location=meta_location, labels_path=labels_path, num_runs=1) #20
# train_stop= perf_counter()
# MLP4_training_time = train_stop - train_start
# print("Malignant vs Benign Re-trained MLP Training Time:", MLP4_training_time)



print(vae_test_loss)
if not math.isnan(vae_test_loss):
#    full_results_list = list(np.load(results_path + '/' + 'Candidate_VAE_and_Classifier_Results.npy', allow_pickle=True))
#    print("[SSIM, Test Loss, MSE, VAE hyperparameters],",
#          "[AUC, Test Loss, Test Accuracy, Performance Metrics: precision, recall, specificity, f1, MLP Hyperparameters]", full_results_list)
     np.save(results_path + '/' + 'run.npy', [Run])
#     results = [metrics_list, Mal_NonMal, Mal_Ben, Mal_NonMal_retrained]
#     full_results_list.append(results)
#     np.save(results_path + '/' + 'Candidate_VAE_and_Classifier_Results.npy', full_results_list)
#     #print("this run:", "[SSIM, Test Loss, MSE, VAE hyperparameters]",
#     #      "[AUC, Test Loss, Test Accuracy, Performance Metrics: precision, recall, specificity, f1, MLP Hyperparameters]")
#     #print(results)

print('VAE Training Time', VAE_training_time)
print("Malignant vs Non-Malignant Re-trained MLP Training Time:", MLP1_training_time)
#print("Malignant vs Benign MLP Training Time:", MLP2_training_time)
print('VAE with Classifier Loss Training Time', VAE_MLP_training_time)
print("Malignant vs Non-Malignant Re-trained MLP Training Time:", MLP3_training_time)

print("malignant vs non-malignant AUC",
      'AUC:', Mal_NonMal[0], 'Test Loss:', Mal_NonMal[1], 'Test Accuracy:', Mal_NonMal[2],
      'Performance Metrics:', Mal_NonMal[3], "Hyperparams", Mal_NonMal[4])

print("malignant vs non-malignant Accuracy",
      'AUC:', Mal_NonMal2[0], 'Test Loss:', Mal_NonMal2[1], 'Test Accuracy:', Mal_NonMal2[2],
      'Performance Metrics:', Mal_NonMal2[3], "Hyperparams", Mal_NonMal2[4])

# print("malignant vs benign AUC",
#       'AUC:', Mal_Ben[0], 'Test Loss:', Mal_Ben[1], 'Test Accuracy:', Mal_Ben[2],
#       'Performance Metrics:', Mal_Ben[3], "Hyperparams", Mal_Ben[4])

# print("malignant vs benign Accuracy",
#       'AUC:', Mal_Ben2[0], 'Test Loss:', Mal_Ben2[1], 'Test Accuracy:', Mal_Ben2[2],
#       'Performance Metrics:', Mal_Ben2[3], "Hyperparams", Mal_Ben2[4])

print("malignant vs non-malignant re-trained AUC",
      'AUC:', Mal_NonMal_retrained[0], 'Test Loss:', Mal_NonMal_retrained[1], 'Test Accuracy:', Mal_NonMal_retrained[2],
      'Performance Metrics:', Mal_NonMal_retrained[3], "Hyperparams", Mal_NonMal_retrained[4])

print("malignant vs non-malignant re-trained Accuracy",
      'AUC:', Mal_NonMal2_retrained[0], 'Test Loss:', Mal_NonMal2_retrained[1], 'Test Accuracy:', Mal_NonMal2_retrained[2],
      'Performance Metrics:', Mal_NonMal2_retrained[3], "Hyperparams", Mal_NonMal2_retrained[4])

# print("malignant vs benign re-trained AUC",
#       'AUC:', Mal_Ben_retrained[0], 'Test Loss:', Mal_Ben_retrained[1], 'Test Accuracy:', Mal_Ben_retrained[2],
#       'Performance Metrics:', Mal_Ben_retrained[3], "Hyperparams", Mal_Ben_retrained[4])

# print("malignant vs benign re-trained Accuracy",
#       'AUC:', Mal_Ben2_retrained[0], 'Test Loss:', Mal_Ben2_retrained[1], 'Test Accuracy:', Mal_Ben2_retrained[2],
#       'Performance Metrics:', Mal_Ben2_retrained[3], "Hyperparams", Mal_Ben2_retrained[4])

print('Mean SSIM Before', VAE1_metrics[0])
print('Mean SSIM After (M v ¬M)', VAE2_metrics[0])
# print('Mean SSIM After (M v B)', VAE3_metrics[0])

print('Mean Squared Error Before', VAE1_metrics[2])
print('Mean Squared Error After (M v ¬M)', VAE2_metrics[2])
# print('Mean Squared Error After (M v B)', VAE3_metrics[2])

print('Mean Absolute Error Before', VAE1_metrics[3])
print('Mean Absolute Error After (M v ¬M)', VAE2_metrics[3])
#print('Mean Absolute Error After (M v B)', VAE3_metrics[3])
