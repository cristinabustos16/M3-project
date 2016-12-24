# Executable program.
from session2 import main
from session2 import general_options_class


# Select options:
options = general_options_class()


# Call main program:
accuracy, running_time = main(options.scale_kmeans, options.apply_pca, options.ncomp_pca, \
                            options.detector_options, options.SVM_options)