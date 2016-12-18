# Main program.
from session1 import SVM_options_class
from session1 import detector_options_class
from session1 import train_and_test


# Select options:
SVM_options = SVM_options_class()
SVM_options.kernel = 'poly'
SVM_options.sigma = 0.01
SVM_options.probability = 1
detector_options = detector_options_class()
detector_options.descriptor = 'ORB'
ncomp_pca = 5
scale = 1
apply_pca = 1

# Call main program:
accuracy, running_time = train_and_test(scale, apply_pca, ncomp_pca, \
    detector_options, SVM_options)

