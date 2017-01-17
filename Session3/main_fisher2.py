# Executable program.
from session3_fisher2 import general_options_class
from session3_fisher2 import main

# Select options:
options = general_options_class()

# Use Fisher Vectors?
options.use_fisher = 1

# Codebook options:
options.compute_codebook = 1
options.fname_codebook = 'codebook512'
options.kmeans = 32

# PCA and scaling:
options.scale_features = 1
options.apply_pca = 1
options.ncomp_pca = 20

# Detector options:
options.detector_options.descriptor = 'SIFT'
options.detector_options.nfeatures = 100

# Dense sampling options:
options.detector_options.dense_sampling = 1
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 8
options.detector_options.dense_sampling_keypoint_radius = 8

# Spatial pyramids options
options.spatial_pyramids = 1
options.spatial_pyramids_depth = 2
options.spatial_pyramids_conf = '3x1'
options.spatial_pyramid_kernel = 0

# Select classifier:
options.classifier = 'svm'

# SVM options:
options.SVM_options.kernel = 'histogramIntersection'
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


#######################################################

# Call the main program:
accuracy, running_time = main(options)




