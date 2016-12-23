import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster

def main():
    start = time.time()

    # read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))

    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)

    # create the SIFT detector object
    SIFTdetector = cv2.SIFT(nfeatures=100)
    
    #extract features from trainning images
    Train_descriptors,D = read_and_extract_features(train_images_filenames,train_labels,SIFTdetector)
    
    k = 512
    codebook = compute_codebook(k,D)

    visual_words = compute_visual_words(Train_descriptors,k,codebook)
    
    stdSlr,clf = train_classifier(visual_words,train_labels)
    
    visual_words_test = test_system(test_images_filenames,SIFTdetector,codebook,k)

    accuracy = 100*clf.score(stdSlr.transform(visual_words_test), test_labels)

    print 'Final accuracy: ' + str(accuracy)

    end=time.time()
    print 'Done in '+str(end-start)+' secs.'

    ## 49.56% in 285 secs.
    return accuracy,end

def read_and_extract_features(train_images_filenames,train_labels,SIFTdetector):

    # extract SIFT keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    Train_descriptors = []
    Train_label_per_descriptor = []
    
    for i in range(len(train_images_filenames)):
        	filename=train_images_filenames[i]
        	print 'Reading image '+filename
        	ima=cv2.imread(filename)
        	gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        	kpt,des=SIFTdetector.detectAndCompute(gray,None)
        	Train_descriptors.append(des)
        	Train_label_per_descriptor.append(train_labels[i])
        	print str(len(kpt))+' extracted keypoints and descriptors'
         
    # Transform everything to numpy arrays
    size_descriptors=Train_descriptors[0].shape[1]
    D=np.zeros((np.sum([len(p) for p in Train_descriptors]),size_descriptors),dtype=np.uint8)
    startingpoint=0
    for i in range(len(Train_descriptors)):
        	D[startingpoint:startingpoint+len(Train_descriptors[i])]=Train_descriptors[i]
        	startingpoint+=len(Train_descriptors[i])

    
    return Train_descriptors,D
    
def compute_codebook(k,D):
    print 'Computing kmeans with '+str(k)+' centroids'
    init=time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=k, verbose=False, batch_size=k * 20,compute_labels=False,reassignment_ratio=10**-4)
    codebook.fit(D)
    cPickle.dump(codebook, open("codebook.dat", "wb"))
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return codebook
    
def compute_visual_words(Train_descriptors,k,codebook):
    init=time.time()
    visual_words=np.zeros((len(Train_descriptors),k),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        words=codebook.predict(Train_descriptors[i])
        visual_words[i,:]=np.bincount(words,minlength=k)

    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    return visual_words
    
def train_classifier(visual_words,train_labels):
    # Train a linear SVM classifier
    stdSlr = StandardScaler().fit(visual_words)
    D_scaled = stdSlr.transform(visual_words)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
    print 'Done!'
    return stdSlr,clf
    
def test_system(test_images_filenames,SIFTdetector,codebook,k):
    # get all the test data and predict their labels
    visual_words_test=np.zeros((len(test_images_filenames),k),dtype=np.float32)
    for i in range(len(test_images_filenames)):
        	filename=test_images_filenames[i]
        	print 'Reading image '+filename
        	ima=cv2.imread(filename)
        	gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        	kpt,des=SIFTdetector.detectAndCompute(gray,None)
        	words=codebook.predict(des)
        	visual_words_test[i,:]=np.bincount(words,minlength=k)
    return visual_words_test