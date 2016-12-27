# Compute and write codebook.
from session2 import compute_and_write_descriptors
from session2 import general_options_class


# Select options:
options = general_options_class()


# Name of the file where to write the codebook (without ending):
fname_descriptors = 'descriptors_SIFT_100'


# Call the program for computing and writing the codebook:
compute_and_write_descriptors(fname_descriptors, options)