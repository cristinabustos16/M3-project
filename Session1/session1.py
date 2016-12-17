import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.decomposition import PCA
import sys


def test_system(test_images_filenames, test_labels, clf, detector, stdSlr, pca, \
                apply_pca, scale):
# Use the test data to measure the performance of the adjusted classifier.
    numtestimages=0
    numcorrect=0
    for i in range(len(test_images_filenames)):
        filename=test_images_filenames[i]
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des=detector.detectAndCompute(gray,None)
        # Scale descriptors:
        if(scale == 1):
            des = stdSlr.transform(des)
        # Apply PCA to descriptors:
        if(apply_pca == 1):
            des = pca.transform(des)
        # Classify:
        predictions = clf.predict(des)
        values, counts = np.unique(predictions, return_counts=True)
        predictedclass = values[np.argmax(counts)]
        print 'image '+filename+' was from class '+test_labels[i]+' and was predicted '+predictedclass
        numtestimages+=1
        if predictedclass==test_labels[i]:
            numcorrect+=1
    accuracy = numcorrect*100.0/numtestimages
    print 'Final accuracy: ' + str(accuracy)
    return accuracy


def train_classifier(X, L, SVM_options):
# Train the classifier, given some options and the training data.
    print 'Training the SVM classifier...'
    sys.stdout.flush()
    if(SVM_options.kernel == 'linear'):
        clf = svm.SVC(kernel='linear', C = SVM_options.C, random_state = 1).fit(X, L)
    elif(SVM_options.kernel == 'poly'):
        clf = svm.SVC(kernel='poly', C = SVM_options.C, degree = SVM_options.degree, coef0 = SVM_options.coef0).fit(X,L)
    elif(SVM_options.kernel == 'rbf'):
        clf = svm.SVC(kernel='rbf', C = SVM_options.C, gamma = SVM_options.sigma, \
                random_state = 1).fit(X, L)
    elif(SVM_options.kernel == 'sigmoid'):
        clf = svm.SVC(kernel='sigmoid', C = SVM_options.C, coef0 = SVM_options.coef0).fit(X, L)
    else:
        print 'SVM kernel not recognized!'
    print 'Done!'
    return clf


def read_and_extract_features(train_images_filenames, train_labels, detector):
# Read the images and extract the features.
    Train_descriptors = []
    Train_label_per_descriptor = []
    for i in range(len(train_images_filenames)):
        filename=train_images_filenames[i]
        if Train_label_per_descriptor.count(train_labels[i])<30:
            print 'Reading image '+filename
            ima=cv2.imread(filename)
            gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
            kpt,des=detector.detectAndCompute(gray,None)
            Train_descriptors.append(des)
            Train_label_per_descriptor.append(train_labels[i])
            print str(len(kpt))+' extracted keypoints and descriptors'
    return Train_descriptors, Train_label_per_descriptor


def train_and_test(scale, apply_pca, ncomp_pca, nfeatures, descriptor, SVM_options):
# Main program, where data is read, features computed, the classifier fit,
# and then applied to the test data.
    start = time.time()

    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))

    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

    # create the SIFT detector object
    if(descriptor=='SIFT'):
        detector = cv2.SIFT(nfeatures)
    elif (descriptor=='SURF'):   
        detector = cv2.SURF(nfeatures)
    else: 
        detector = cv2.ORB(nfeatures)
    # read the just 30 train images per class
    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    Train_descriptors, Train_label_per_descriptor = \
        read_and_extract_features(train_images_filenames, train_labels, detector)

    # Transform everything to numpy arrays
    D=Train_descriptors[0]
    L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
    for i in range(1,len(Train_descriptors)):
        D=np.vstack((D,Train_descriptors[i]))
        L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))

    # Scale input data, and keep the scaler for later:
    stdSlr = StandardScaler().fit(D)
    if(scale == 1):
        D = stdSlr.transform(D)

    # PCA:
    pca = PCA(n_components = ncomp_pca)
    if(apply_pca == 1):
        print "Applying principal components analysis..."
        pca.fit(D)
        D = pca.transform(D)
        print "Explained variance with ", ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'

    # Train a linear SVM classifier
    clf = train_classifier(D, L, SVM_options)

    # get all the test data and predict their labels
    accuracy = test_system(test_images_filenames, test_labels, clf, detector, \
        stdSlr, pca, apply_pca, scale)
    
    end=time.time()
    running_time = end-start
    print 'Done in '+str(running_time)+' secs.'
    
    # Return accuracy and time:
    return accuracy, running_time
    

class SVM_options_class:
# Options for SVM classifier.
    kernel = 'poly'
    C = 1
    sigma = 1
    degree = 3
    coef0 = 0

