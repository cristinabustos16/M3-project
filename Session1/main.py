# Main program.
from session1 import SVM_options_class
from session1 import train_and_test


# Select options:
SVM_options = SVM_options_class()
SVM_options.kernel = 'rbf'
SVM_options.sigma = 0.001
ncomp_pca = 50
nfeatures = 100
scale = 1
apply_pca = 1
detector = 'SIFT'

# Call main program:
accuracy, running_time = train_and_test(scale, apply_pca, ncomp_pca, \
    nfeatures, detector, SVM_options)

