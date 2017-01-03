# Compute and write codebook.
from session2 import compute_and_write_codebook
from session2 import general_options_class


# Select options:
options = general_options_class()
options.codebook = 512

# Apply dense sampling to the selected detector
options.detector_options.dense_sampling = 1
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 10
options.detector_options.dense_sampling_keypoint_radius = 5

# Name of the file where to write the codebook (without ending):
filename = 'codebook512_dense'


# Call the program for computing and writing the codebook:
compute_and_write_codebook(options, filename)