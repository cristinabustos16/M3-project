#### Get the results from the random search, and look for the best one.
import os
import cPickle
import numpy as np

# Directory for saving results:
dirResults = './random_search/'
# dirResults = './random_search_nodropout/'
    
# Load previous results:
cases_done = [] # Initialize list with done cases.
# List previous files:
previous_files = os.listdir(dirResults)
nprevious = len(previous_files)    
print str(nprevious) + ' previous cases found.'
if nprevious > 0:
    # Get the data from the previous cases:
    for filename in previous_files:
        case = cPickle.load(open(dirResults + filename, 'r'))
        parameters = [] # List with parameters of a given case.
        for i in range(len(case)):
            parameters.append(case[i][1])
        cases_done.append(parameters)

# Loop over the cases found and plot from min to max accuracy:

accuracy_array = []
for i in range(len(cases_done)):
    case = cases_done[i]
    accuracy = case[len(case)-1]
    accuracy_array.append(accuracy)

index_sorted = np.argsort(accuracy_array)

for i in range(len(index_sorted)):
    index = index_sorted[i]
    print 'Best parameters found:'
    print 'batch_size = ' + str(cases_done[index][0])
    print 'nepochs = ' + str(cases_done[index][1])
    print 'optimizer_name = ' + str(cases_done[index][2])
    print 'learn_rate = ' + str(cases_done[index][3])
    print 'momentum = ' + str(cases_done[index][4])
    print 'dropout_probability = ' + str(cases_done[index][5])
    print 'accuracy = ' + str(cases_done[index][6])
    print '\n' + '********************************************'