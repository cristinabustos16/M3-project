# Launching program with several options for SVM.
from session1 import SVM_options_class
from session1 import train_and_test
from joblib import Parallel, delayed

from math import sqrt


# General options, common in al calls:
ncomp_pca = 20
SIFT_nfeatures = 100
scale = 1
apply_pca = 1

# SVM options:
SVM_options = []
for i in range(4):
    instance = SVM_options_class()
    SVM_options.append(instance)
# Case 1:
SVM_options[0] = SVM_options_class()
# Case 2:
SVM_options[1] = SVM_options_class()
SVM_options[1].kernel = 'rbf'
SVM_options[1].sigma = 1
# Case 3:
SVM_options[2] = SVM_options_class()
SVM_options[2].kernel = 'rbf'
SVM_options[2].sigma = 0.5
# Case 4:
SVM_options[3] = SVM_options_class()
SVM_options[3].kernel = 'rbf'
SVM_options[3].sigma = 2



# Call main program:
for i in range(4):
    accuracy, running_time = train_and_test(scale, apply_pca, ncomp_pca, \
        SIFT_nfeatures, SVM_options[i])
    filename = 'resultados' + str(i) + '.txt'
    fid = open(filename, 'w')
    line = 'accuracy = ' + str(accuracy) + '    running_time = ' + str(running_time)
    fid.write(line)
    fid.close()







