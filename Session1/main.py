# Main program.
from session1 import SVM_options_class
from session1 import train_and_test


# Select options:
SVM_options = SVM_options_class()
ncomp_pca = 20
nfeatures = 100
scale = 1
apply_pca = 1
detector = 'SIFT'

# Call main program:
accuracy, running_time = train_and_test(scale, apply_pca, ncomp_pca, \
    nfeatures, detector, SVM_options)

## 38.78% in 797 secs.
## With 20 principal components: 30.86% in 362 sec.
## With 50 principal components: 36.18% in 583 sec.