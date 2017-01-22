# Executable program.
import time
import cPickle
import numpy as np
import sys
from session4 import general_options_class
from session4 import create_subsets_cross_validation
from session4 import train_system_cnn_SVM
from session4 import test_system_cnn_SVM
from session4 import train_system_cnn
from session4 import test_system_cnn
from session4 import create_cnn
# Select options:
options = general_options_class()

# Codebook options:
options.compute_codebook = 1
options.kmeans = 64

# PCA and scaling:
options.scale_features = 1
options.apply_pca = 0
options.ncomp_pca = 100

# Use Fisher Vectors?
options.use_fisher = 1

# Cross-validation options:
options.compute_subsets = 1
options.k_cv = 5

# SVM options:
options.SVM_options.kernel = 'linear'
options.SVM_options.C = 1
options.SVM_options.sigma = 0.01
options.SVM_options.degree = 3
options.SVM_options.coef0 = 0
options.SVM_options.probability = 1

# Evaluation options:
options.compute_evaluation = 0
options.system = 'SVM'

#######################################################

start = time.time()

# Create the cross-validation subsets:
if options.compute_subsets:
    create_subsets_cross_validation(options.k_cv)
    
# Read the subsets:
# We create a list where the element i is another list with all the file
# names belonging to subset i. The same for the labels of the images.
subsets_filenames = list(xrange(options.k_cv))
subsets_labels = list(xrange(options.k_cv))
# Loop over the subsets:
for i in range(options.k_cv):
    subsets_filenames[i] = cPickle.load(open('subset_'+str(i)+'_filenames.dat','r'))
    subsets_labels[i] = cPickle.load(open('subset_'+str(i)+'_labels.dat','r'))

# Initialize vector to store the accuracy of each training.
accuracy = np.zeros(options.k_cv)

# Train and evaluate k times:
for i in range(options.k_cv):
    # First, we create a list with the names of the files we will use for
    # training. These are the files in all the subsets except the i-th one.
    # The same for the labels.
    # Initialize:
    trainset_images_filenames = []
    trainset_labels = []
    for j in range(options.k_cv):
        if(i != j): # If it is not the i-th subset, add it to our trainset.
            trainset_images_filenames.extend(subsets_filenames[j])
            trainset_labels.extend(subsets_labels[j])
    # For validation, we will use the rest of the images, i.e., the
    # subset i.
    validation_images_filenames = subsets_filenames[i]
    validation_labels = subsets_labels[i]

    if options.system == 'SVM':
        cnn, clf, stdSlr, pca = train_system_cnn_SVM(trainset_images_filenames, trainset_labels, options)
        accuracy[i] = test_system_cnn_SVM(validation_images_filenames, validation_labels, cnn, stdSlr, pca, clf, options)
    elif options.system == 'BoW':
        
        detector = create_cnn('block5_conv2')
        clf, codebook, stdSlr_VW, stdSlr_features, pca = train_system_cnn(trainset_images_filenames, trainset_labels, detector, options)
        accuracy[i] = test_system_cnn(validation_images_filenames, validation_labels, \
                                    detector, codebook, clf, stdSlr_VW, \
                                    stdSlr_features, pca, options)
        
# Compute the mean and the standard deviation of the accuracies found:
accuracy_mean = np.mean(accuracy)
print('Mean accuracy: ' + str(accuracy_mean))
accuracy_sd = np.std(accuracy, ddof = 1)
print('Std. dev. accuracy: ' + str(accuracy_sd))

end = time.time()
running_time = end-start
print 'Done in '+str(running_time)+' secs.'

# Write the results in a text file:
report_name = 'report_' + options.file_name + '.txt'
fd = open(report_name, 'w')
try:
    fd.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
    fd.write('\n' + 'Std. dev. accuracy: ' + str(accuracy_sd))
    fd.write('\n' + 'Done in ' + str(end - start) + ' secs.')
except OSError:
    sys.stdout.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
fd.close()