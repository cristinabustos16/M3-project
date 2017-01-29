#### Get the results from the random search, and look for the best one.
import os
import cPickle

# Directory for saving results:
#dirResults = './random_search/'
dirResults = './random_search/'
    
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

# Loop over the cases found, finding the best accuracy:
best_case = 0
max_accuracy = 0
for i in range(len(cases_done)):
    case = cases_done[i]
    accuracy = case[len(case)-1]
    if(max_accuracy < accuracy):
        best_case = i
        max_accuracy = accuracy
print 'Best parameters found:'
print 'batch_size = ' + str(cases_done[best_case][0])
print 'nepochs = ' + str(cases_done[best_case][1])
print 'optimizer_name = ' + str(cases_done[best_case][2])
print 'learn_rate = ' + str(cases_done[best_case][3])
print 'momentum = ' + str(cases_done[best_case][4])
print 'accuracy = ' + str(accuracy)