# VAE_lung_lesion_BMVC
# Variational Autoencoders for Feature Exploration and Malignancy Prediction of Lung Lesions
Exploration of LIDC-IDRI lung lesion dataset (https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=1966254)

# Contents:

## Preprocessing
#### *LIDC_DICOM_to_Numpy.ipynb*
* Converts the CT scans from DICOM format to numpy arrays (numerical vectors for Houndsfield Units of pixel intensities) and crop images to a region of interest (ROI) 64x64 pixels. Includes calculations for how to implement the bounding box cropping of the image based on the centre of the region annotated in the segmentation masks.

#### *LIDC_datasplit.ipynb*
* View the metadata given with the LIDC-IDRI dataset and saves malignancy labels to a numpy array. Includes removal of the slices from the labels which were excluded.

#### *mask_size.ipynb*
* Find the region of interest (ROI) size for the lesions based on the convex hull and minimum bounding box of the segmentation masks. 

#### *Train_Test_Split.ipynb*  
* Split the patients into train/validation/test it produces the following two files also saved here *meta_mal_ben.csv* and *meta_mal_nonmal.csv*. These files hold the meta-data for both splits of the patients: malignant vs benign (mal_ben) with ambiguous excluded and malignant vs non-malignant (mal_nonmal).

## VAE
#### *Extract Latent Vectors and Reconstructions.ipynb*
* Extract the latent vectors from the VAE model using the model parameters saved and save the latent vectors.

#### *RandomSearchVAE.py*  
* Gaussian VAE with hyperparameter training combined with MLP predictor to assess calssification quality of latent vectors. Note: includes splitting slices at patient level.
  
#### *RandomSearch_Dirichlet_VAE.py*  
* VAE with Dirichlet latent space. Note: produces latent vectors with better disentanglement which may allow better latent exploration as each dimension in latent vector is encouraged to encode different features.  

#### *VAE_MLP_joint_loss_mal_nonmal.py*
* Gaussian VAE malignant vs non-malignant with joint VAE and classifier loss.

#### *VAE_joint_loss_mal_benign.py*
* Gaussian VAE malignant vs benign with joint VAE and classifier loss.

#### *VAE_Dirichlet_joint_loss.py*:
* Dirichlet VAE with joint VAE and classifier loss.

  
## Clusering and MLP
### Files used in MSc
#### *Clustering_inital.ipynb*
* This file explores clustering of the latent vectors. Including extracting latent vectors, exploration using PCA and t-SNE and k-means clustering.

#### *Clustering.ipynb*
* Grid search for best clustering of K-Means and CLASSIX (https://github.com/nla-group/classix).

#### *Exploration_gaussian.ipynb*
* Latent space exploration and code to generate latent traversal figures.

#### *RandomSearchMLP.py* 
* This file does a larger random hyperparameter search than my other random search files (in VAE). This script runs cross-validation on the latent vectors to find the best results of the classifier.

#### *Dirichlet_RandomSearchMLP.py*
* This file does the random hyperparameter search for the Dirichlet VAE
