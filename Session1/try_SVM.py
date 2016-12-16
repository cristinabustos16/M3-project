# Run the code with a certain configuration for the SVM, and write the results.
# This way, executing this script separately at the same time, with different
# options, the SVM tunning can be performed in parallel.
from session1 import SVM_options_class
from session1 import train_and_test
import sys


# Select options:
SVM_options = SVM_options_class()
ncomp_pca = 20
SIFT_nfeatures = 100
scale = 1
apply_pca = 1
    
# Read file name for writing results:
filename = sys.argv[1]

# Read options for SVM:
SVM_options.kernel = sys.argv[2]
SVM_options.C = float(sys.argv[3])
if SVM_options.kernel == 'rbf':
    SVM_options.sigma = float(sys.argv[4])

# Call main program:
accuracy, running_time = train_and_test(scale, apply_pca, ncomp_pca, \
    SIFT_nfeatures, SVM_options)

# Write results:
fid = open(filename, 'w')
line1 = 'kernel = ' + SVM_options.kernel + ', C = ' + str(SVM_options.C) + \
    ', sigma = ' + str(SVM_options.sigma) + '\n'
line2 = 'accuracy = ' + str(accuracy) + '    running_time = ' + str(running_time)
fid.write(line1)
fid.write(line2)
fid.close()