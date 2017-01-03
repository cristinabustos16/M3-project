# Executable program.
from session2 import train_and_evaluate
from session2 import general_options_class


# Select options:
options = general_options_class()
options.k_cv = 5 # Number of subsets for cross-validation.


# Call main program:
accuracy_mean, accuracy_sd = train_and_evaluate(options)