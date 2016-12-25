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
    
    # Extract features from train images:
    D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector)
    
    # Fit the scaler and the PCA:
    stdSlr_kmeans, pca = preprocess_fit(D, scale_kmeans, apply_pca, ncomp_pca)
    
    # Scale and apply PCA to features:
    D = preprocess_apply(D, scale_kmeans, apply_pca, stdSlr_kmeans, pca, ncomp_pca)
    
    # Compute the codebook with the features:
    k = 512
    codebook = compute_codebook(k, D)

    # Cast features to Bag of Visual Words:
    visual_words = descriptors2words(D, codebook, k, descriptors_per_image)
    
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(visual_words)
    
    # Scale words:
    visual_words = stdSlr_VW.transform(visual_words)
    
    # Train the classifier:
    clf = train_classifier(visual_words, train_labels, SVM_options)
    
    # Test de system:
    accuracy = test_system(test_images_filenames, detector, codebook, k, test_labels, \
                            clf, stdSlr_VW, scale_kmeans, stdSlr_kmeans, apply_pca, \
                            ncomp_pca, pca)

    print 'Final accuracy: ' + str(accuracy)
    end = time.time()
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


def read_and_extract_features(images_filenames, detector):
    # extract keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    descriptors = []
    nimages = len(images_filenames)
    descriptors_per_image = np.zeros(nimages, dtype=np.uint8)
    for i in range(nimages):
        filename = images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des = detector.detectAndCompute(gray,None)
        descriptors_per_image[i] = len(kpt)
        descriptors.append(des)
        print str(descriptors_per_image[i]) + ' extracted keypoints and descriptors'
    
    # Transform everything to numpy arrays
    size_descriptors = descriptors[0].shape[1]
    D = np.zeros((np.sum([len(p) for p in descriptors]), size_descriptors), dtype=np.float64)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint += len(descriptors[i])
            
    return D, descriptors_per_image
    
    
def preprocess_fit(D, scale_kmeans, apply_pca, ncomp_pca):
    # Fit the scaler and the PCA with the training features.
    stdSlr_kmeans = StandardScaler()
    if(scale_kmeans == 1):
        stdSlr_kmeans = StandardScaler().fit(D)
    pca = PCA(n_components = ncomp_pca)
    if(apply_pca == 1):
        print "Applying principal components analysis..."
        pca.fit(D)
        print "Explained variance with ", ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'
    return stdSlr_kmeans, pca
    
    
def preprocess_apply(D, scale_kmeans, apply_pca, stdSlr_kmeans, pca, ncomp_pca):
    # Scale and apply PCA to features.
    if(scale_kmeans == 1):
        D = stdSlr_kmeans.transform(D)
    if(apply_pca == 1):
        D = pca.transform(D)
    return D
    
    
def compute_codebook(k, D):
    # Clustering (unsupervised classification)
    # Apply kmeans over the features to computer the codebook.
    print 'Computing kmeans with '+str(k)+' centroids'
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, \
            batch_size = k * 20, compute_labels=False, reassignment_ratio=10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return codebook
    
    
#def read_codebook(codebook_file):
    # Read the codebook from the specified file.
    #return codebook
    
    
def descriptors2words(D, codebook, k, descriptors_per_image):
    # Compute the visual words, given the features and a codebook.
    nimages = len(descriptors_per_image)
    visual_words = np.zeros((nimages,k), dtype=np.float32)
    ini = 0
    for i in xrange(nimages):
        fin = ini + descriptors_per_image[i]
        words = codebook.predict(D[ini:fin,:])
        visual_words[i,:] = np.bincount(words, minlength=k)
        ini = fin
    return visual_words


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
                        
    # Extract features form test images:
    D, descriptors_per_image = read_and_extract_features(test_images_filenames, detector)
    
    # Scale and apply PCA to the extracted features:
    D = preprocess_apply(D, scale_kmeans, apply_pca, stdSlr_kmeans, pca, ncomp_pca)

    # Cast features to visual words:
    visual_words_test = descriptors2words(D, codebook, k, descriptors_per_image)
    
    # Scale visual words:
    visual_words_scaled = stdSlr_VW.transform(visual_words_test)
    
    # Compute accuracy:
    accuracy = 100 * clf.score(visual_words_scaled, test_labels)

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