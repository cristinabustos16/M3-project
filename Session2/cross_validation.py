# Executable program.
from session2 import train_and_evaluate
from session2 import general_options_class


# Select options:
options = general_options_class()
options.scale_kmeans = 0
options.apply_pca = 0
options.ncomp_pca = 50
options.k_cv = 5
options.compute_codebook = 1
options.fname_codebook = 'codebook512'
options.compute_descriptors = 1
options.fname_descriptors = 'descriptors_SIFT_100'


# Call main program:
accuracy_mean, accuracy_sd = train_and_evaluate(options)