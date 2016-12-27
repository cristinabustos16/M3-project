# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()
options.scale_kmeans = 0
options.apply_pca = 0
options.ncomp_pca = 50


# Name of the file with the codebook (without ending):
fname_codebook = 'codebook512'
fname_descriptors = 'descriptors_SIFT_100'


# Call main program:
accuracy, running_time = main(options, fname_codebook, fname_descriptors)