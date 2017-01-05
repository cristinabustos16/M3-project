# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()

options.compute_codebook = 0
options.fname_codebook = 'codebook512'

options.spatial_pyramids = 0
options.depth = 3

# Detector options:
options.detector_options.descriptor = 'SIFT'
options.detector_options.nfeatures = 100
# Apply dense sampling to the selected detector
options.detector_options.dense_sampling = 0
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 10
options.detector_options.dense_sampling_keypoint_radius = 5

# Filename to identify the test
# file_name will be added to the saved plots and the report text file
file_name = "desc_%s_dense_%s_kmeans_%s_pyramids_%s_kernel_%s_C_%s_sigma_%s" \
            % (options.detector_options.descriptor, options.detector_options.dense_sampling, options.kmeans, \
               options.spatial_pyramids, options.SVM_options.kernel, options.SVM_options.C, options.SVM_options.sigma)

options.file_name = file_name
options.show_plots = 0 #Don't save the plots (Show plots blocks the program execution)
options.save_plots = 1 #save plots as png with the file_name provided (roc_curve_filename, conf_matrix_filename,
                      # prec_recall_filename)
# Call main program:
accuracy, running_time = main(options)