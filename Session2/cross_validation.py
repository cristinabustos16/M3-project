# Executable program.
from session2 import train_and_evaluate
from session2 import general_options_class


# Select options:
options = general_options_class()
options.k_cv = 5 # Number of subsets for cross-validation.
options.plot_confusion_matrix = 0 # It's a mess this with cross-validation.

options.compute_codebook = 0 # Compute or read the codebook.
options.fname_codebook = 'codebook512' # In case of reading the codebook, specify here the name of the file.

options.spatial_pyramids = 1
options.depth = 2

# Apply dense sampling to the selected detector
options.detector_options.dense_sampling = 0
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 10
options.detector_options.dense_sampling_keypoint_radius = 5


# Call main program:
accuracy_mean, accuracy_sd = train_and_evaluate(options)





