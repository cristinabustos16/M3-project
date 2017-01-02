import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from sklearn.decomposition import PCA
import sys


##############################################################################
def train_and_evaluate(options):

    # Create the detector object
    detector = create_detector(options.detector_options)
    
    accuracy = np.zeros(options.k_cv)
    
    # Read the subsets:
    subsets_filenames = list(xrange(options.k_cv))
    subsets_labels = list(xrange(options.k_cv))
    for i in range(options.k_cv):
        subsets_filenames[i] = cPickle.load(open('subset_'+str(i)+'_filenames.dat','r'))
        subsets_labels[i] = cPickle.load(open('subset_'+str(i)+'_labels.dat','r'))
        
    for i in range(options.k_cv):
        trainset_filenames = []
        trainset_labels = []
        for j in range(options.k_cv):
            if(i != j):
                trainset_filenames.extend(subsets_filenames[j])
                trainset_labels.extend(subsets_labels[j])
        validation_filenames = subsets_filenames[i]
        validation_labels = subsets_labels[i]
    
        # Train system:
        clf, codebook, stdSlr_VW, stdSlr_kmeans, pca = \
                                    train_system(trainset_filenames, \
                                    trainset_labels, detector, options)
    
        # Evaluate system:
        accuracy[i] = test_system(validation_filenames, validation_labels, \
                                detector, codebook, clf, stdSlr_VW, \
                                stdSlr_kmeans, pca, options)
           
    # Compute the mean and the standard deviation of the accuracies found:
    accuracy_mean = np.mean(accuracy)
    accuracy_sd = np.std(accuracy, ddof = 1)
                     
    return accuracy_mean, accuracy_sd


##############################################################################
def main(options):
    start = time.time()

    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))

    print 'Loaded ' + str(len(train_images_filenames)) + \
            ' training images filenames with classes ',set(train_labels)
    print 'Loaded ' + str(len(test_images_filenames)) + \
            ' testing images filenames with classes ',set(test_labels)

    # Create the detector object
    detector = create_detector(options.detector_options)
    
    # Train system:
    clf, codebook, stdSlr_VW, stdSlr_kmeans, pca = \
                                    train_system(train_images_filenames, \
                                    train_labels, detector, options)
    
    # Test system:
    accuracy = test_system(test_images_filenames, test_labels, detector, codebook, \
                            clf, stdSlr_VW, stdSlr_kmeans, pca, options)

    print 'Final accuracy: ' + str(accuracy)
    end = time.time()
    print 'Done in '+str(end-start)+' secs.' 
    
    return accuracy, end


##############################################################################
def create_subsets_cross_validation(k_cv):
    # Create a split for k-fold Cross-Validation.
    # Read the whole training set:
    train_images_filenames = \
                np.array(cPickle.load(open('train_images_filenames.dat','r')))
    train_labels = np.array(cPickle.load(open('train_labels.dat','r')))
    
    nimages = len(train_labels)
    nimages_persubset = np.divide(nimages, k_cv)
    
    # Sample random numbers in the range of 0 to the number of images, without
    # replacement. This is like shuffeling the indexes.
    shuffled_images = np.random.choice(range(nimages), nimages, replace = False)
    
    ini = 0
    for i in range(k_cv):
        if i == k_cv-1:
            fin = nimages
        else:
            fin = ini + nimages_persubset
        # Indexes for subset i:
        subset = shuffled_images[ini:fin]
        # From indexes to filenames and labels:
        subset_filenames = list(train_images_filenames[subset])
        subset_labels = list(train_labels[subset])
#        subset_filenames = train_images_filenames[subset]
#        subset_labels = train_labels[subset]
        # Write the subsets:
        cPickle.dump(subset_filenames, open('subset_'+str(i)+'_filenames.dat', "wb"))
        cPickle.dump(subset_labels, open('subset_'+str(i)+'_labels.dat', "wb"))
#        np.savetxt('subset_'+str(i)+'_filenames.txt', subset_filenames, fmt='%s')
#        np.savetxt('subset_'+str(i)+'_labels.txt', subset_labels, fmt='%s')
        # Update beginning of indexes:
        ini = fin
        
    
##############################################################################
def compute_and_write_codebook(options, fname_codebook):
    
    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))

    print 'Loaded ' + str(len(train_images_filenames)) + \
            ' training images filenames with classes ',set(train_labels)

    # Create the detector object
    detector = create_detector(options.detector_options)
    
    # Extract features from train images:
    D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector)
    
    # Fit the scaler and the PCA:
    stdSlr_kmeans, pca = preprocess_fit(D, options)
    
    # Scale and apply PCA to features:
    D = preprocess_apply(D, stdSlr_kmeans, pca, options)
    
    # Compute the codebook with the features:
    codebook = compute_codebook(options.kmeans, D)
    
    # Write codebook:
    cPickle.dump(codebook, open(fname_codebook+'.dat', "wb"))
    
    
##############################################################################
def compute_and_write_descriptors(fname_descriptors, options):
    # Save the descriptors computed over the images, stored in a matrix D,
    # as well as the number of descriptors that each image has.

    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    # Create the detector object
    detector = create_detector(options.detector_options)
    # Extract features from train images:
    D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector, options.detector_options)
    # Write arrays:
    np.savetxt(fname_descriptors + '_D.txt', D, fmt = '%u')
    np.savetxt(fname_descriptors + '_dpi.txt', descriptors_per_image, fmt = '%u')
    
    
##############################################################################
def create_detector(detector_options):
    # create the detector object
    if(detector_options.descriptor == 'SIFT'):
        detector = cv2.SIFT(detector_options.nfeatures)
    elif (detector_options.descriptor == 'SURF'):   
        detector = cv2.SURF(detector_options.SURF_hessian_ths)
    elif (detector_options.descriptor == 'ORB'):   
        detector = cv2.ORB(detector_options.nfeatures)
    else: 
        print 'Error: feature detector not recognized.'
    return detector


def dense_sampling(max_nr_keypoints, step_size, radius, image_height, image_width):

    nr_keypoints = (image_height/step_size)*(image_width/step_size)
    while not nr_keypoints <= max_nr_keypoints:
        step_size = step_size - 1
        nr_keypoints = (image_height / step_size) * (image_width / step_size)

    if step_size < 1:
        step_size = 1

    kpt = [cv2.KeyPoint(x, y, radius) for y in range(step_size-1, image_height-step_size, step_size)
                                        for x in range(step_size-1, image_width-step_size, step_size)]

    #kpt = [cv2.KeyPoint(x, y, radius) for y in range(0, image_height, step_size)
     #      for x in range(0, image_width, step_size)]

    return kpt


##############################################################################
def read_and_extract_features(images_filenames, detector, detector_options):
    # extract keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    descriptors = []
    nimages = len(images_filenames)
    descriptors_per_image = np.zeros(nimages, dtype=np.uint16)
    for i in range(nimages):
        filename = images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if detector_options.dense_sampling == 1:
            kpt = dense_sampling(detector_options.dense_sampling_max_nr_keypoints, detector_options.dense_sampling_keypoint_step_size, \
                                 detector_options.dense_sampling_keypoint_radius, gray.shape[0], gray.shape[1])
            #kpt, des = detector.detectAndCompute(gray, None, useProvidedKeypoints=True)
            des = detector.compute(gray, kpt)
        else:
            kpt,des = detector.detectAndCompute(gray,None)
        descriptors_per_image[i] = len(kpt)
        descriptors.append(des)
        print str(descriptors_per_image[i]) + ' extracted keypoints and descriptors'
    
    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.float32)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint += len(descriptors[i])
            
    return D, descriptors_per_image
    
    
##############################################################################
def preprocess_fit(D, options):
    # Fit the scaler and the PCA with the training features.
    stdSlr_kmeans = StandardScaler()
    if(options.scale_kmeans == 1):
        stdSlr_kmeans = StandardScaler().fit(D)
    pca = PCA(n_components = options.ncomp_pca)
    if(options.apply_pca == 1):
        print "Applying principal components analysis..."
        pca.fit(D)
        print "Explained variance with ", options.ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'
    return stdSlr_kmeans, pca
    
    
##############################################################################
def preprocess_apply(D, stdSlr_kmeans, pca, options):
    # Scale and apply PCA to features.
    if(options.scale_kmeans == 1):
        D = stdSlr_kmeans.transform(D)
    if(options.apply_pca == 1):
        D = pca.transform(D)
    return D
    
    
##############################################################################
def compute_codebook(kmeans, D):
    # Clustering (unsupervised classification)
    # Apply kmeans over the features to computer the codebook.
    print 'Computing kmeans with ' + str(kmeans) + ' centroids'
    sys.stdout.flush()
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=kmeans, verbose=False, \
            batch_size = kmeans * 20, compute_labels=False, \
            reassignment_ratio=10**-4, random_state = 1)
    codebook.fit(D)
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return codebook
    
    
##############################################################################
def read_codebook(fname_codebook):
    # Read the codebook from the specified file.
    with open(fname_codebook+'.dat', "rb") as input_file:
        codebook = cPickle.load(input_file)
    return codebook
    
    
##############################################################################
def read_descriptors(fname_descriptors):
    D = np.loadtxt(fname_descriptors + '_D.txt', dtype = np.float32)
    descriptors_per_image = np.loadtxt(fname_descriptors + '_dpi.txt', dtype = np.uint8)
    return D, descriptors_per_image
    
    
##############################################################################
def descriptors2words(D, codebook, kmeans, descriptors_per_image):
    # Compute the visual words, given the features and a codebook.
    nimages = len(descriptors_per_image)
    visual_words = np.zeros((nimages,kmeans), dtype=np.float32)
    ini = 0
    for i in xrange(nimages):
        fin = ini + descriptors_per_image[i]
        words = codebook.predict(D[ini:fin,:])
        visual_words[i,:] = np.bincount(words, minlength=kmeans)
        ini = fin
    return visual_words


##############################################################################
def train_classifier(X, L, SVM_options):
# Train the classifier, given some options and the training data.
    print 'Training the SVM classifier...'
    sys.stdout.flush()
    if(SVM_options.kernel == 'linear'):
        clf = svm.SVC(kernel='linear', C = SVM_options.C, random_state = 1, \
                probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'poly'):
        clf = svm.SVC(kernel='poly', C = SVM_options.C, degree = SVM_options.degree, \
                coef0 = SVM_options.coef0, random_state = 1, \
                probability = SVM_options.probability).fit(X,L)
    elif(SVM_options.kernel == 'rbf'):
        clf = svm.SVC(kernel='rbf', C = SVM_options.C, gamma = SVM_options.sigma, \
                random_state = 1, probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'sigmoid'):
        clf = svm.SVC(kernel='sigmoid', C = SVM_options.C, coef0 = SVM_options.coef0, \
                random_state = 1, probability = SVM_options.probability).fit(X, L)
    else:
        print 'SVM kernel not recognized!'
    print 'Done!'
    return clf
    
    
##############################################################################
def train_system(train_images_filenames, train_labels, detector, options):
    # Train the system with the training data.
    
    # Getting the image descriptors:
    if options.compute_descriptors:                
        # Extract features from train images:
        D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector, options.detector_options)
    else:
        # Read descriptors:
        D, descriptors_per_image = read_descriptors(options.fname_descriptors)
        
    # Getting the codebook:
    if options.compute_codebook:
        # Compute the codebook:
        codebook = compute_codebook(options.kmeans, D)
    else:
        # Read codebook:
        codebook = read_codebook(options.fname_codebook)
    
    # Fit the scaler and the PCA:
    stdSlr_kmeans, pca = preprocess_fit(D, options)
    
    # Scale and apply PCA to features:
    D = preprocess_apply(D, stdSlr_kmeans, pca, options)

    # Cast features to Bag of Visual Words:
    visual_words = descriptors2words(D, codebook, options.kmeans, descriptors_per_image)
    
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(visual_words)
    
    # Scale words:
    visual_words = stdSlr_VW.transform(visual_words)
    
    # Train the classifier:
    clf = train_classifier(visual_words, train_labels, options.SVM_options)
    
    return clf, codebook, stdSlr_VW, stdSlr_kmeans, pca
    
    
##############################################################################
def test_system(test_images_filenames, test_labels, detector, codebook, clf, \
                    stdSlr_VW, stdSlr_kmeans, pca, options):
    # get all the test data and predict their labels
                        
    # Extract features form test images:
    D, descriptors_per_image = read_and_extract_features(test_images_filenames, detector)
    
    # Scale and apply PCA to the extracted features:
    D = preprocess_apply(D, stdSlr_kmeans, pca, options)

    # Cast features to visual words:
    visual_words_test = descriptors2words(D, codebook, options.kmeans, descriptors_per_image)
    
    # Scale visual words:
    visual_words_scaled = stdSlr_VW.transform(visual_words_test)
    
    # Compute accuracy:
    accuracy = 100 * clf.score(visual_words_scaled, test_labels)

    return accuracy
    

##############################################################################
class SVM_options_class:
# Options for SVM classifier.
    kernel = 'linear'
    C = 1
    sigma = 1
    degree = 3
    coef0 = 0
    probability = 0 # This changes the way we aggregate predictions.
    

##############################################################################
class detector_options_class:
# Options feature detectors.
    descriptor = 'SIFT'
    nfeatures = 100
    SURF_hessian_ths = 400
    dense_sampling = 0  # Apply dense sampling to the selected detector
    dense_sampling_max_nr_keypoints = 50000  # Maximum number of equally spaced keypoints
    dense_sampling_keypoint_step_size = 5
    dense_sampling_keypoint_radius = 10  # Maximum number of equally spaced keypoints
    

##############################################################################
class general_options_class:
# General options for the system.
    SVM_options = SVM_options_class()
    detector_options = detector_options_class()
    ncomp_pca = 30 # Number of components for PCA.
    scale_kmeans = 0 # Scale features befores applying kmeans.
    apply_pca = 0 # Apply, or not, PCA.
    kmeans = 512 # Number of cluster for k-means (codebook).
    k_cv = 5 # Number of subsets for k-fold cross-validation.
    compute_codebook = 1 # Compute or read the codebook.
    fname_codebook = 'codebook512' # In case of reading the codebook, specify here name of the file.
    compute_descriptors = 1 # Compute or read the image descriptors.
    fname_descriptors = 'descriptors_SIFT_100' # In case of reading the descriptors, specify here name of the file.