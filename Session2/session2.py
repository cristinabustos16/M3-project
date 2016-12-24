import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from sklearn.decomposition import PCA
import sys


def main(scale_kmeans, apply_pca, ncomp_pca, detector_options, SVM_options):
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

    # create the detector object
    detector = create_detector(detector_options)
    
    #extract features from trainning images
    Train_descriptors, D, stdSlr_kmeans, pca = \
                        read_and_extract_features(train_images_filenames, \
                        train_labels, detector, scale_kmeans, apply_pca, ncomp_pca)
    
    k = 512
    codebook = compute_codebook(k, D)

    visual_words, stdSlr_VW = compute_visual_words(Train_descriptors, k, codebook)
    
    clf = train_classifier(visual_words, train_labels, SVM_options)
    
    accuracy = test_system(test_images_filenames, detector, codebook, k, test_labels, \
                            clf, stdSlr_VW, scale_kmeans, stdSlr_kmeans, apply_pca, \
                            ncomp_pca, pca)

    print 'Final accuracy: ' + str(accuracy)
    end=time.time()
    print 'Done in '+str(end-start)+' secs.'
    return accuracy, end
    
    
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


def read_and_extract_features(train_images_filenames, train_labels, detector, \
                        scale_kmeans, apply_pca, ncomp_pca):
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    Train_descriptors = []
    Train_label_per_descriptor = []
    for i in range(len(train_images_filenames)):
        	filename = train_images_filenames[i]
        	print 'Reading image ' + filename
        	ima = cv2.imread(filename)
        	gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        	kpt,des = detector.detectAndCompute(gray,None)
        	Train_descriptors.append(des)
        	Train_label_per_descriptor.append(train_labels[i])
        	print str(len(kpt)) + ' extracted keypoints and descriptors'
         
    # Transform everything to numpy arrays
    size_descriptors = Train_descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in Train_descriptors]), size_descriptors), dtype=np.uint8)
    startingpoint = 0
    for i in range(len(Train_descriptors)):
        	D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
        	startingpoint += len(Train_descriptors[i])

    # Scale input data, and keep the scaler for later:
    stdSlr_kmeans = StandardScaler().fit(D)
    if(scale_kmeans == 1):
        D = stdSlr_kmeans.transform(D)

    # PCA:
    pca = PCA(n_components = ncomp_pca)
    if(apply_pca == 1):
        print "Applying principal components analysis..."
        pca.fit(D)
        D = pca.transform(D)
        print "Explained variance with ", ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'
            
    return Train_descriptors, D, stdSlr_kmeans, pca
    
    
def compute_codebook(k, D):
    print 'Computing kmeans with '+str(k)+' centroids'
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, \
            batch_size=k * 20, compute_labels=False, reassignment_ratio=10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return codebook
    
    
def compute_visual_words(Train_descriptors, k, codebook):
    init = time.time()
    visual_words = np.zeros((len(Train_descriptors),k),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        words = codebook.predict(Train_descriptors[i])
        visual_words[i,:] = np.bincount(words,minlength=k)
    stdSlr_VW = StandardScaler().fit(visual_words)
    visual_words = stdSlr_VW.transform(visual_words)
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return visual_words, stdSlr_VW


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
    
    
def test_system(test_images_filenames, detector, codebook, k, test_labels, clf, stdSlr_VW, \
                        scale_kmeans, stdSlr_kmeans, apply_pca, ncomp_pca, pca):
    # get all the test data and predict their labels
    visual_words_test = np.zeros((len(test_images_filenames),k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        filename = test_images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt, des = detector.detectAndCompute(gray, None)
        # Scale descriptors:
        if(scale_kmeans == 1):
            des = stdSlr_kmeans.transform(des)
        # Apply PCA to descriptors:
        if(apply_pca == 1):
            des = pca.transform(des)
        words = codebook.predict(des)
        visual_words_test[i,:] = np.bincount(words, minlength=k)
    accuracy = 100 * clf.score(stdSlr_VW.transform(visual_words_test), test_labels)
    return accuracy
    

class SVM_options_class:
# Options for SVM classifier.
    kernel = 'linear'
    C = 1
    sigma = 1
    degree = 3
    coef0 = 0
    probability = 0 # This changes the way we aggregate predictions.
    

class detector_options_class:
# Options feature detectors.
    descriptor = 'SIFT'
    nfeatures = 100
    SURF_hessian_ths = 400
    

class general_options_class:
# General options for the system.
    SVM_options = SVM_options_class()
    detector_options = detector_options_class()
    ncomp_pca = 30 # Number of components for PCA.
    scale_kmeans = 0 # Scale features befores applying kmeans.
    apply_pca = 0 # Apply, or not, PCA.