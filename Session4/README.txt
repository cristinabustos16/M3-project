M3 - Machine Learning for Computer Vision.
Project documentation for session 4.
22/01/2017

Team 3:
    - Lidia Garrucho Moras
    - Xenia Salinas Ventalló
    - María Cristina Bustos Rodríguez
    - Xián López Álvarez

There are two possibilities to execute the code:
To train with the whole train set and evaluate with the test set, use main.py
To make cross-validation, use cross_validation.py.

All the functions are in the script session4.py, which are called from main.py and cross_validation.py.

Below are listed some of the functionalities of the system, with the corresponding option that enables it (1 for enabling, 0 for disabling):
To use the output of a convolutional layer as features for BoW or Fisher, use “system = ‘BoW’”; otherwise, to directly classify with the output of the last FC layer of the CNN, switch “system = ‘FC’”.
Select layer from which to extract features:    “layer_cnn_bow”
Fisher Vectors:          “use_fisher”
PCA:                     “apply_pca”
Scale before PCA:        “scale_features”

More options are available, which are described at the end of file session4.py.

List of main files:
    - session4.py: Code of the system.
    - main.py: Specify the options, train the system with the whole training set, and then evaluate with the test set.
    - cross_validation.py: Specify the options, and perform a cross-validation of the system. The mean and the standard deviation of the accuracies will be returned.
