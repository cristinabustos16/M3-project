M4 - Machine Learning for Computer Vision.
Project documentation for session 2.
01/01/2017

Team 3:
	- Lidia Garrucho Moras
	- Xenia Salinas Ventalló
	- María Cristina Bustos Rodríguez
	- Xián López Álvarez

List of main files:
	- session2.py: Contains the functions for the tasks of the week, which are called from other scripts.
	- main.py: Specify the options, train the system with the whole training set, and then evaluate with
						the test set.
	- cross_validation.py: Specify the options, and perform a cross-validation of the system. The mean and
						the standard deviation of the accuracies will be returned.
	- write_codebook.py: Must be redesigned. The idea is to compute the codebook, and then store store it, 
						so we don't have to compute it again all the time.

Structure of the code:
	- We have created a class for the options of the system. This is the only input the system requires, so
		everything is specified through this. Another classes for the options of the SVM and the feature
		detector exist, but they are contained in the general options as well.