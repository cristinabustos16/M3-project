M3 - Machine Learning for Computer Vision.
Project documentation for session 2.
17/01/2017

Team 3:
    - Lidia Garrucho Moras
    - Xenia Salinas Ventalló
    - María Cristina Bustos Rodríguez
    - Xián López Álvarez

There are two possibilities to execute the code:
To train with the whole train set and evaluate with the test set, use main.py
To make cross-validation, use cross_validation.py.

All the functions are in the script session3.py, which are called from main.py and cross_validation.py.

For Fisher Vectors, since they need the library yael, which is not available en Windows, we created a duplicate of the code, in the scripts that have the suffix “_fisher”.

Below are listed some of the functionalities of the system, with the corresponding option that enables it (1 for enabling, 0 for disabling):
Dense SIFT:            	“detector_options.dense_sampling”
Spatial pyramid:        “spatial_pyramids”.
ROC & conf. mat.:	“compute_evaluation”
Fisher Vectors:		“use_fisher”
PCA:                	“apply_pca”
Scale features:        	“scale_features”

More options are available, which are described at the end of file session3.py, for things like choosing the configuration of the spatial pyramid, its depth, the number of components of PCA, the configuration of dense sampling, etc...

List of main files:
    - session3.py: Code of the system.
    - main.py: Specify the options, train the system with the whole training set, and then evaluate with the test set.
    - cross_validation.py: Specify the options, and perform a cross-validation of the system. The mean and the standard deviation of the accuracies will be returned.