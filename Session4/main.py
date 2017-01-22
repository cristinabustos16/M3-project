# Executable program.
from session4 import general_options_class
from session4 import main_cnn
from session4 import main_cnn_SVM

# Select options:
options = general_options_class()

# Use CNN to extract features:
options.features_from_cnn = 1

# Layer from which to extract the features (for BoW with CNN):
options.layer_cnn_bow = 'block5_conv2'

# Use Fisher Vectors?
options.use_fisher = 0

# Codebook options:
options.compute_codebook = 1
options.kmeans = 512

# PCA and scaling:
options.scale_features = 0
options.apply_pca = 0
options.ncomp_pca = 100

# Alternative to PCA:
options.aggregate_alt_pca = 'mean-max-min' # ('none', 'mean', 'max', 'mean-max', 'mean-max-min')

# Use Fisher Vectors?
options.use_fisher = 0

# Select classifier:
options.classifier = 'svm'

# SVM options:
options.SVM_options.kernel = 'linear'
options.SVM_options.C = 1
options.SVM_options.sigma = 1
options.SVM_options.degree = 3
options.SVM_options.coef0 = 0
options.SVM_options.probability = 1

# Evaluation options:
options.compute_evaluation = 0

# Reduce dataset?
options.reduce_dataset = 1

# Classify output of FC layers (SVM), or use BoW (BoW):
options.system = 'BoW'


#######################################################

# Call the main program:
if options.system == 'SVM':
    accuracy, running_time = main_cnn_SVM(options)
elif options.system == 'BoW':
    accuracy, running_time = main_cnn(options)




