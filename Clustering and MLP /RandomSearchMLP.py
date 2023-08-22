candidate_num = 40
print("candidate num", candidate_num)
import numpy as np

# PyTorch
import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision import transforms
from torch.utils.data import ConcatDataset, random_split, DataLoader
from sklearn import metrics

import pandas as pd
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from time import perf_counter
import time
import random

job_start_time = perf_counter()

save_results_path = r"VAE_lung_lesion/MLP"
meta_location = r"VAE_lung_lesion/Preprocessing"


# Check if GPU available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

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
latent_vectors = np.load(r"/nobackup/mm17b2k/VAE_RandomSearch/latent_vectors_{}.npy".format(candidate_num))
#latent_vectors2 = np.load(r"VAE_lung_lesion/MLP/latent_vectors/mu2original.npy")
ambiguous = np.load(r"VAE_lung_lesion/MLP/latent_vectors/ambiguous.npy")
latent_vectors2 = []
for i, vec in enumerate(latent_vectors):
    if i not in ambiguous:
        latent_vectors2.append(torch.Tensor(vec))
latent_vectors2 = torch.stack(latent_vectors2)
#np.save(r"VAE_lung_lesion/MLP/latent_vectors/mu2.npy", latent_vectors2)

input_size = latent_vectors.shape[1]
input_size2 = latent_vectors2.shape[1]

print("latent vector size:", input_size)

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
    
    split_indicies = torch.tensor(list(meta["data_split"]))
    print(split_indicies.shape)
    print(split_indicies)
    #cross_val_data_test = [torch.index_select(latent_vectors, 0, split_indicies)]
    #cross_val_labels_test = [torch.index_select(labels, 0, split_indicies)]
    
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
    #print("data shape test",cross_val_data_test.shape, "labels shape test", cross_val_labels_test.shape)
    print("data shape",len(cross_val_data), "labels shape", len(cross_val_labels))
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

    print("The confusion matrix is:")
    print(cm)
    print(cm2)
    print("The accuracy is ", np.trace(cm) / np.sum(cm) * 100)
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
    #print('Precision,', 'proprotion of malignant predictions that are true:', precision)
    recall = tp/(tp+fn)
    #print('Recall,', 'proportion of tumours identified:', recall)
    specificity = tn/(tn+fp)
    #print('Specificity,', 'proportion of non-cancerous lesions identified:', specificity)
    f1 = 2*((precision*recall)/(precision+recall))
    #print('F1 score:', f1)
    
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
        if counter % 5 == 0:
            print('early stopping counter at:', counter)
      #  if counter >= tolerance:
      #        return True
    return counter

def train_model(nepochs, train_loader, valid_loader, test_loader, hyperparams):
    global input_size, results_path
    threshold = hyperparams["threshold"]
    lr = hyperparams["lr"]
    size1 = hyperparams["layer_sizes"][0]
    size2 = hyperparams["layer_sizes"][1]
    size3 = hyperparams["layer_sizes"][2]
    dropout = hyperparams["dropout"]
    Depth = hyperparams["Depth"]
    
    if Depth == 4:
        model = nn.Sequential(
            nn.Linear(input_size,size1),
            nn.BatchNorm1d(size1, eps=0.0001, momentum=0.3),
            nn.Dropout(dropout),
            nn.GELU(),

            nn.Linear(size1,size2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),

            nn.Linear(size2,size3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size3, eps=0.0001, momentum=0.3),

            nn.Linear(size3,1),
            nn.Sigmoid())

    if Depth == 5:
        model = nn.Sequential(
            nn.Linear(input_size,size1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size1, eps=0.0001, momentum=0.3),

            nn.Linear(size1,size2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),

            nn.Linear(size2,size2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size2, eps=0.0001, momentum=0.3),


            nn.Linear(size2,size3),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(size3, eps=0.0001, momentum=0.3),


            nn.Linear(size3,1),
            nn.Sigmoid())

    
    model.to(device)
    results_path = r"VAE_lung_lesion/MLP/MLP.pt"
    statsrec = np.zeros((4,nepochs))

    loss_fn = nn.BCELoss()
    optimiser = optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=20, 
                                                               threshold=0.01, threshold_mode='abs')
    counter = 0
    for epoch in range(1,nepochs+1):  # loop over the dataset multiple times
        correct = 0                   # number of examples predicted correctly (for accuracy)
        total = 0                     # number of examples
        running_loss = 0.0            # accumulated loss (for mean loss)
        n = 0                         # number of minibatches
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
        if epoch % 75 == 0 or epoch == 1 or epoch == nepochs - 1 or counter == 25:
            print(f"epoch: {epoch} training loss: {ltrn: .3f} training accuracy: {atrn: .1%}  validation loss: {lval: .3f} validation accuracy: {aval: .1%}")
            
        test_results = stats(test_loader,model, threshold)
        test_loss, test_acc = test_results[0], test_results[1]
        test_outputs, test_labels = test_results[2], test_results[3]
        if epoch % 75 == 0 or epoch == nepochs - 1 or counter == 25:
            print('test loss:', test_loss.item(),'test accuracy:', test_acc*100, '%')  
        scheduler.step(ltrn)
        counter = early_stopping(counter, ltrn, lval, min_delta=0.4)
        if counter > 25:
            print("At Epoch:", epoch)
            break
    # save network parameters, losses and accuracy
    torch.save({"stats": statsrec}, results_path) #"state_dict": model.state_dict()
    #plot_results(results_path, "fulltrainset.jpg", epoch)
    test_cm = confusion_matrix(test_outputs, test_labels, threshold)
    auc = metrics.roc_auc_score(test_labels, test_outputs)
    print('AUC is:', auc)
    results = evaluation_metrics(test_cm[1,1], test_cm[0,1], test_cm[0,0], test_cm[1,0])
    
    return test_loss.item(), test_acc, results, auc


def test_hyperparams(parameter_space, data):
    global latent_vectors2, job_start_time
    accuracy_list, loss_list, auc_list, hyperparams_list, results_list, best_results = [], [], [], [], [], []
    results_path = r"/nobackup/mm17b2k/VAE_RandomSearch"
    
    if data == 1:
        # malignant vs non-malignant
        meta = pd.read_csv(meta_location + '//' + 'meta_mal_nonmal.csv')
        latent_vectors = np.load(results_path + '/' + "latent_vectors_{}.npy".format(candidate_num), allow_pickle=True)
        labels = np.load(r"VAE_lung_lesion/MLP/latent_vectors/labels2.npy")
    if data == 2:
        # malignant vs benign
        meta = pd.read_csv(meta_location + '//' + 'meta_mal_ben.csv')
        latent_vectors = latent_vectors2 #np.load(r"VAE_lung_lesion/MLP/latent_vectors/mu2original.npy")
        labels = np.load(r"VAE_lung_lesion/MLP/latent_vectors/labels3.npy")

        
    i = 0
    mlp_runs, num_tried, stop_counter = 0, 0, 0
    while len(hyperparams_list) < 55:
        mlp_runs +=1
        print("Attempt:", mlp_runs)
        print("Number of completed attempts:", num_tried)
        # if job time is near 48 hours
        current_job_time = perf_counter()
        if current_job_time - job_start_time > 60*60*47.8:
            print("running out of time for job (at 47hrs 48 mins)")
            break
        if data == 1 and current_job_time - job_start_time > 60*60*24:
            print("running out of time for job (part 1) (at 24hrs)")
            break

        i+=1
        print(i)
        random.seed(time.time())
        hyperparams = {}
        hyperparams["threshold"] = random.choice(parameter_space["threshold"])
        hyperparams["lr"] = random.choice(parameter_space["lr"])
        hyperparams["layer_sizes"] = random.choice(parameter_space["layer_sizes"])
        hyperparams["dropout"] = random.choice(parameter_space["dropout"])
        batch_size = random.choice(parameter_space["batch_size"])
        hyperparams["batch_size"] = batch_size
        hyperparams["Depth"] = random.choice(parameter_space["Depth"])     
                    
        if hyperparams in hyperparams_list:
            print("Already tried these hyperparams")
            continue
        
        print(hyperparams)

        
        train_start = perf_counter()
        CV_loss, CV_accuracy, CV_results, CV_auc = [], [], [], []
        for run in range(5):  # 5-fold cross validation
            train_loader, valid_loader, test_loader = Cross_Validation(run, 5, meta, latent_vectors, labels, batch_size)
            loss, accuracy, results, auc = train_model(300, train_loader, valid_loader, test_loader, hyperparams)
            if accuracy < 0.675:
                print("########## This is not a good candidate for cross validation ##########")
                print("Number of completed attempts so far:", num_tried)
                break       
                
            CV_loss.append(loss)
            CV_accuracy.append(accuracy)
            CV_results.append(results)
            CV_auc.append(auc)
            
          
        train_stop= perf_counter()
        print('training time', train_stop - train_start)  
       
        if len(CV_auc) < 5:
            stop_counter += 1
            
        if stop_counter == 100:
            print("Break due to 100 failed attempts")
            break
        
        if len(CV_auc) == 5:
            print(".................Cross Validation Averages.................")
            print("AUC:", np.mean(CV_auc), "Loss:", np.mean(CV_loss), "Accuracy:", np.mean(CV_accuracy), "Results:", average_metrics(CV_results))
            auc_list.append(np.mean(CV_auc))
            loss_list.append(np.mean(CV_loss))
            accuracy_list.append(np.mean(CV_accuracy))
            hyperparams_list.append(hyperparams)
            results_list.append(average_metrics(CV_results))
            best_results.append([max(CV_auc), min(CV_auc), max(CV_accuracy), min(CV_accuracy)])
            num_tried +=1
            print("Completed Attempt:", num_tried, "Total Tried", i) 
            
            auc_list = np.nan_to_num(auc_list).tolist()
            idx = auc_list.index(max(auc_list))
            print("Best so far", "Based on AUC:",
                 'AUC:', max(auc_list), 
                 'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
                 'Performance Metrics:', results_list[idx],'Index:', idx,
                 'Hyperparameters:', hyperparams_list[auc_list.index(max(auc_list))],
                 'Maximum and Minimum AUC and Accuracy:', best_results[idx])
            
            accuracy_list = np.nan_to_num(accuracy_list).tolist()   
            idx2 = accuracy_list.index(max(accuracy_list))
            print("Best so far", "Based on Accuracy",
                 'AUC:', auc_list[idx2], 'Test Loss:', loss_list[idx2], 'Test Accuracy:', accuracy_list[idx2], 
                 'Performance Metrics:', results_list[idx2],'Index:', idx2,
                 'Hyperparameters:', hyperparams_list[accuracy_list.index(max(accuracy_list))],
                 'Maximum and Minimum AUC and Accuracy:', best_results[idx2])
    accuracy_list = np.nan_to_num(accuracy_list).tolist()  
    idx = auc_list.index(max(auc_list))
    if len(auc_list) > 1:
        print("Based on AUC:",
          'AUC:', max(auc_list),
          'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx],
          'Performance Metrics:', results_list[idx],'Index:', idx, 
          'Hyperparameters:', hyperparams_list[auc_list.index(max(auc_list))],
          'Maximum and Minimum AUC and Accuracy:', best_results[idx])
    accuracy_list = np.nan_to_num(accuracy_list).tolist()
    idx = accuracy_list.index(max(accuracy_list))
    if len(accuracy_list) > 1:
        print("Based on Accuracy",
          'AUC:', auc_list[idx], 'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
          'Performance Metrics:', results_list[idx],'Index:', idx, 
          'Hyperparameters:', hyperparams_list[accuracy_list.index(max(accuracy_list))],
          'Maximum and Minimum AUC and Accuracy:', best_results[idx])
     
    np.save(save_results_path + '/' + 'auc_list_{}.npy'.format(data), auc_list) 
    np.save(save_results_path + '/' + 'loss_list_{}.npy'.format(data), loss_list) 
    np.save(save_results_path + '/' + 'accuracy_list_{}.npy'.format(data), accuracy_list) 
    np.save(save_results_path + '/' + 'hyperparams_list_{}.npy'.format(data), hyperparams_list)
    np.save(save_results_path + '/' + 'metrics_list_{}.npy'.format(data), results_list) 
    np.save(save_results_path + '/' + 'best_results_{}.npy'.format(data), best_results) 

    return auc_list, loss_list, accuracy_list, hyperparams_list, best_results


parameter_space = {"threshold":[0.6,0.55,0.5,0.45,0.4], "lr":[1e-6,1e-5,5e-5], 
                   "layer_sizes":[[2048, 2048, 1024], [2048, 1024, 512], [2048, 1024, 256], [2048, 512, 512],
                                  [2048, 512, 256], [2048, 512, 128], [1024, 1024, 512], [1024, 1024, 256],
                                  [1024, 512, 512], [1024, 512, 256], [1024, 256, 256], [512, 512, 256],
                                  [512, 256, 256]],
                   "dropout":[0.2,0.25,0.4,0.5,0.55], "batch_size":[32,64,128,256,512],
                   "Depth":[4,5]
                  }

test_hyperparams(parameter_space, data=1)

test_hyperparams(parameter_space, data=2)


auc_list = list(np.load(save_results_path + '/' + 'auc_list_1.npy', allow_pickle=True))
loss_list = list(np.load(save_results_path + '/' + 'loss_list_1.npy', allow_pickle=True)) 
accuracy_list = list(np.load(save_results_path + '/' + 'accuracy_list_1.npy', allow_pickle=True))
hyperparams_list = list(np.load(save_results_path + '/' + 'hyperparams_list_1.npy', allow_pickle=True))
results_list = list(np.load(save_results_path + '/' + 'metrics_list_1.npy', allow_pickle=True))
best_results = list(np.load(save_results_path + '/' + 'best_results_1.npy', allow_pickle=True))

auc_list = np.nan_to_num(auc_list).tolist()
idx = auc_list.index(max(auc_list))
#if idx != 0 and len(auc_list) > 1:
print("Based on AUC",
      "malignant vs non-malignant",
      'AUC:', max(auc_list), 'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
      'Performance Metrics:', results_list[idx],'Index:', idx, 
      'Hyperparameters:', hyperparams_list[auc_list.index(max(auc_list))],
      'Maximum and Minimum AUC and Accuracy (max min, max min):', best_results[idx])
accuracy_list = np.nan_to_num(accuracy_list).tolist()
idx = accuracy_list.index(max(accuracy_list))
#if idx != 0 and len(accuracy_list) > 1:
print("Based on Accuracy",
      "malignant vs non-malignant",
      'AUC:', auc_list[idx], 'Test Loss:', loss_list[idx], 'Test Accuracy:', accuracy_list[idx], 
      'Performance Metrics:', results_list[idx],'Index:', idx, 
      'Hyperparameters:', hyperparams_list[accuracy_list.index(max(accuracy_list))],
      'Maximum AUC and Accuracy:', best_results[idx])
