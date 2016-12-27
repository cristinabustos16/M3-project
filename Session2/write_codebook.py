# Compute and write codebook.
from session2 import compute_and_write_codebook
from session2 import general_options_class


# Select options:
options = general_options_class()
options.scale_kmeans = 1
options.apply_pca = 0
options.ncomp_pca = 50
options.codebook = 512


# Name of the file where to write the codebook (without ending):
filename = 'codebook512_scale'


# Call the program for computing and writing the codebook:
compute_and_write_codebook(options, filename)