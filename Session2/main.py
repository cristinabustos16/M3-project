# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()
options.scale_kmeans = 0
options.apply_pca = 0
options.ncomp_pca = 50
options.compute_codebook = 1
options.fname_codebook = 'codebook512'
options.compute_descriptors = 1
options.fname_descriptors = 'descriptors_SIFT_100'


# Call main program:
accuracy, running_time = main(options)