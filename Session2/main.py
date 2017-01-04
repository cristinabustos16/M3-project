# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()

options.compute_codebook = 0
options.fname_codebook = 'codebook512'

options.spatial_pyramids = 1
options.depth = 3

# Detector options:
options.detector_options.descriptor = 'SIFT'
options.detector_options.nfeatures = 100
# Apply dense sampling to the selected detector
options.detector_options.dense_sampling = 1
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 10
options.detector_options.dense_sampling_keypoint_radius = 5

# Call main program:
accuracy, running_time = main(options)