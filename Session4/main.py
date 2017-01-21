# Executable program.
from session4 import general_options_class
from session4 import main_cnn
from session4 import main_cnn_SVM

# Select options:
options = general_options_class()

# Use CNN to extract features:
options.features_from_cnn = 1

# Codebook options:
options.compute_codebook = 1
options.kmeans = 512

# PCA and scaling:
options.scale_features = 1
options.apply_pca = 1
options.ncomp_pca = 100

# Select classifier:
options.classifier = 'svm'

# SVM options:
options.SVM_options.kernel = 'linear'
options.SVM_options.C = 1
options.SVM_options.sigma = 1
options.SVM_options.degree = 3
options.SVM_options.coef0 = 0
options.SVM_options.probability = 1

# Random Forest options:

# Adaboost options:

# Evaluation options:
options.compute_evaluation = 0
options.save_plots = 0
options.file_name = 'test_kernel_'
options.show_plots = 1

# Reduce dataset?
options.reduce_dataset = 1

options.system = 'SVM'


#######################################################

# Call the main program:
if options.system == 'SVM':
    accuracy, running_time = main_cnn_SVM(options)
elif options.system == 'BoW':
    accuracy, running_time = main_cnn(options)




