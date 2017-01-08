# Executable program.
import cPickle
import time
from session2 import general_options_class
from session2 import read_codebook
from session2 import create_detector
from session2 import train_system
from session2 import test_system

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

# read the train and test files
train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
train_labels = cPickle.load(open('train_labels.dat','r'))
test_labels = cPickle.load(open('test_labels.dat','r'))

print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

#Read codebook
codebook = read_codebook(options.fname_codebook)

# Create the detector object
detector = create_detector(options.detector_options)

if not options.spatial_pyramids:
    options.depth = 1
    
clf,stdSlr_VW = train_system(train_images_filenames, train_labels, detector, codebook, options)
accuracy = test_system(test_images_filenames, test_labels, detector, codebook, clf, stdSlr_VW, options)

end=time.time()
running_time = end-start
print 'Accuracy: ' + str(accuracy) + '%'
print 'Done in '+str(running_time)+' secs.'