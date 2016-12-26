# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()
options.scale_kmeans = 0
options.apply_pca = 0


# Name of the file with the codebook (without ending):
filename = 'codebook512'


# Call main program:
accuracy, running_time = main(options, filename)