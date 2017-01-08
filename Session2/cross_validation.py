# Executable program.
import cPickle
import numpy as np
import sys
import time
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from session2 import general_options_class
from session2 import compute_and_write_codebook
from session2 import read_codebook
from session2 import create_subsets_cross_validation
from session2 import create_detector
from session2 import read_and_extract_visual_words
from session2 import train_classifier
from session2 import compute_and_save_confusion_matrix
from session2 import compute_and_save_roc_curve
from session2 import compute_and_save_precision_recall_curve
from sklearn.metrics import classification_report

# Select options:
options = general_options_class()

#Codebook options
options.compute_codebook = 0
options.fname_codebook = 'codebook512'
options.kmeans = 512

#Corss-validation options
options.compute_subsets = 0
options.k_cv = 5

# Detector options:
options.detector_options.descriptor = 'SIFT'
options.detector_options.nfeatures = 100
#Dense sampling options
options.detector_options.dense_sampling = 0
# Maximum number of equally spaced keypoints (Grid size)
options.detector_options.dense_sampling_max_nr_keypoints = 1500
options.detector_options.dense_sampling_keypoint_step_size = 8
options.detector_options.dense_sampling_keypoint_radius = 8
#Spatial pyramids options
options.spatial_pyramids = 0
options.depth = 3

#SVM options
options.SVM_options.kernel = 'histogramIntersection'
options.SVM_options.C = 1.5
options.SVM_options.sigma = 1
options.SVM_options.degree = 3
options.SVM_options.coef0 = 0
options.SVM_options.probability = 1

#Evaluation options
options.file_descriptor = open('report.txt', 'w')
options.save_plots = 1
options.plot_name = 'test'
options.file_name = 'test_kernel_'
options.show_plots = 1

start = time.time()

#Compute or read the codebook
if options.compute_codebook:
    codebook = compute_and_write_codebook(options)
else:
    codebook = read_codebook(options.fname_codebook)
    
#Create the cross-validation subsets
if options.compute_subsets:
    create_subsets_cross_validation(options.k_cv)
    
# Read the subsets:
# We create a list where the element i is another list with all the file
# names belonging to subset i. The same for the labels of the images.
subsets_filenames = list(xrange(options.k_cv))
subsets_labels = list(xrange(options.k_cv))
# Loop over the subsets:
for i in range(options.k_cv):
    subsets_filenames[i] = cPickle.load(open('subset_'+str(i)+'_filenames.dat','rb'))
    subsets_labels[i] = cPickle.load(open('subset_'+str(i)+'_labels.dat','rb'))
    
# Create the detector object
detector = create_detector(options.detector_options)

if not options.spatial_pyramids:
    options.depth = 1

# Extract the features for all the subsets
subset_visual_words = list(xrange(options.k_cv))
# Loop over the subsets:
for i in range(options.k_cv):
    subset_visual_words[i] = read_and_extract_visual_words(subsets_filenames[i], detector, codebook, options)
    
# Initialize vector to store the accuracy of each training.
accuracy = np.zeros(options.k_cv)

# Train and evaluate k times:
for i in range(options.k_cv):
    # First, we create a list with the names of the files we will use for
    # training. These are the files in all the subsets except the i-th one.
    # The same for the labels.
    # Initialize:
    trainset_visual_words = []
    trainset_labels = []
    for j in range(options.k_cv):
        if(i != j): # If it is not the i-th subset, add it to our trainset.
            trainset_visual_words.extend(subset_visual_words[j])
            trainset_labels.extend(subsets_labels[j])
    # For validation, we will use the rest of the images, i.e., the
    # subset i.
    validation_visual_words = subset_visual_words[i]
    validation_labels = subsets_labels[i]

    # The rest is exactly the same as a normal training-testing: we train
    # with the trainset we have just build, and test with the evaluation
    # set.

    # Train system:
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(trainset_visual_words)
    
    # Scale words:
    trainset_visual_words_scaled = stdSlr_VW.transform(trainset_visual_words)
    
    # Train the classifier:
    clf = train_classifier(trainset_visual_words_scaled, trainset_labels, options.SVM_options)

    # Evaluate system:
    validation_visual_words_scaled = stdSlr_VW.transform(validation_visual_words)
    predictions = clf.predict(validation_visual_words_scaled)
    accuracy[i] = 100 * clf.score(validation_visual_words_scaled, validation_labels)
    # Only if pass a valid file descriptor
    if options.file_descriptor != -1:
        options.plot_name = options.file_name + '_' + str(i)
        target_names = ['class mountain', 'class inside_city', 'class Opencountry', 'class coast', 'class street', \
                    'class forest', 'class tallbuilding', 'class highway']
        options.file_descriptor.write(classification_report(validation_labels, predictions, target_names=target_names))
        # Confussion matrix:
        compute_and_save_confusion_matrix(validation_labels, predictions, options)

        classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
        # Compute probabilities:
        predicted_probabilities = clf.predict_proba(validation_visual_words_scaled)
        predicted_score = clf.decision_function(validation_visual_words_scaled)
        # Binarize the labels
        binary_labels = label_binarize(validation_labels, classes=classes)
        # Compute ROC curve and ROC area for each class
        compute_and_save_roc_curve(binary_labels, predicted_probabilities, classes, options)
        # Compute Precision-Recall curve for each class
        compute_and_save_precision_recall_curve(binary_labels, predicted_score, classes, options)
        
options.file_descriptor.close()

end = time.time()
       
# Compute the mean and the standard deviation of the accuracies found:
accuracy_mean = np.mean(accuracy)
print('Mean accuracy: ' + str(accuracy_mean))
accuracy_sd = np.std(accuracy, ddof = 1)
print('Std accuracy: ' + str(accuracy_sd))

# Write the results in a text file
report_name = 'report_' + options.file_name + '.txt'
fd = open(report_name, 'w')
try:
    fd.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
    fd.write('\n' + 'Std accuracy: ' + str(accuracy_sd))
    fd.write('\n' + 'Done in ' + str(end - start) + ' secs.')
except OSError:
    sys.stdout.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
fd.close()

print 'Done in '+str(end-start)+' secs.'

