# Executable program.
from session3 import general_options_class
from session3 import train_and_validate
from session3 import train_and_validate_slow

# Select options:
options = general_options_class()

# Fast or slow cross-validation.
options.fast_cross_validation = 0

# Codebook options:
options.compute_codebook = 0
options.kmeans = 512

# PCA and scaling:
options.scale_features = 0
options.apply_pca = 0
options.ncomp_pca = 60

# Cross-validation options:
options.compute_subsets = 0
options.k_cv = 5

# Detector options:
options.detector_options.descriptor = 'SIFT'
options.detector_options.nfeatures = 100

# Dense sampling options:
options.detector_options.dense_sampling = 1
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 8
options.detector_options.dense_sampling_keypoint_radius = 8

# Spatial pyramids options:
options.spatial_pyramids = 1
options.spatial_pyramids_depth = 2
options.spatial_pyramids_conf = '3x1'

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


#######################################################

# Call the cross-validation program:
if options.fast_cross_validation == 1:
    accuracy_mean, accuracy_sd, running_time = train_and_validate(options)
else:
    accuracy_mean, accuracy_sd, running_time = train_and_validate_slow(options)





