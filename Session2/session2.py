import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
from sklearn.decomposition import PCA
import sys
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize

##############################################################################
def train_and_evaluate(options):

    # Create the detector object
    detector = create_detector(options.detector_options)
    
    # Initialize vector to store the accuracy of each training.
    accuracy = np.zeros(options.k_cv)
    
    # Read the subsets:
    # We create a list where the element i is another list with all the file
    # names belonging to subset i. The same for the labels of the images.
    subsets_filenames = list(xrange(options.k_cv))
    subsets_labels = list(xrange(options.k_cv))
    # Loop over the subsets:
    for i in range(options.k_cv):
        subsets_filenames[i] = cPickle.load(open('subset_'+str(i)+'_filenames.dat','r'))
        subsets_labels[i] = cPickle.load(open('subset_'+str(i)+'_labels.dat','r'))
    
    # Train and evaluate k times:
    for i in range(options.k_cv):
        # First, we create a list with the names of the files we will use for
        # training. These are the files in all the subsets except the i-th one.
        # The same for the labels.
        # Initialize:
        trainset_filenames = []
        trainset_labels = []
        for j in range(options.k_cv):
            if(i != j): # If it is not the i-th subset, add it to our trainset.
                trainset_filenames.extend(subsets_filenames[j])
                trainset_labels.extend(subsets_labels[j])
        # For validation, we will use the rest of the images, i.e., the
        # subset i.
        validation_filenames = subsets_filenames[i]
        validation_labels = subsets_labels[i]
        
        # The rest is exactly the same as a normal training-testing: we train
        # with the trainset we have just build, and test with the evaluation
        # set.
    
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
    D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector, options.detector_options)
    
    # Fit the scaler and the PCA:
    stdSlr_kmeans, pca = preprocess_fit(D, options)
    
    # Scale and apply PCA to features:
    D = preprocess_apply(D, stdSlr_kmeans, pca, options)
    
    # Compute the codebook with the features:
    codebook = compute_codebook(options.kmeans, D)
    
    # Write codebook:
    cPickle.dump(codebook, open(fname_codebook+'.dat', "wb"))
    
    
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


##############################################################################
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
    descriptors_per_image = np.zeros(nimages, dtype=np.uint)
    for i in range(nimages):
        filename = images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if detector_options.dense_sampling == 1:
            kpt = dense_sampling(detector_options.dense_sampling_max_nr_keypoints, detector_options.dense_sampling_keypoint_step_size, \
                                 detector_options.dense_sampling_keypoint_radius, gray.shape[0], gray.shape[1])
            kpt, des = detector.compute(gray, kpt)
            #descriptors_per_image[i] = kpt.__len__()
        else:
            kpt, des = detector.detectAndCompute(gray,None)
        descriptors_per_image[i] = kpt.__len__()
        #descriptors_per_image[i] = len(kpt)
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
    with open(fname_codebook+'.dat', "r") as input_file:
        codebook = cPickle.load(input_file)
    return codebook
    
    
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


    if options.spatial_pyramids:
    
        # Getting the codebook:
        if options.compute_codebook:
            # Compute the codebook:
            print 'Error: not possible to compute codebook with spatial pyramids.'
            sys.exit()
        else:
            # Read codebook:
            codebook = read_codebook(options.fname_codebook)
    
        visual_words = read_and_extract_visual_words(train_images_filenames, \
                    detector, codebook, options)
                    
        stdSlr_kmeans = 0
        pca = 0
        
        
    else:
        # Extract features from train images:
        D, descriptors_per_image = read_and_extract_features(train_images_filenames, \
                    detector, options.detector_options)
    
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
                        
    if options.spatial_pyramids:
        visual_words_test = read_and_extract_visual_words(test_images_filenames, \
                detector, codebook, options)
    
    else:
        # Extract features form test images:
        D, descriptors_per_image = read_and_extract_features(test_images_filenames, detector, options.detector_options)
        
        # Scale and apply PCA to the extracted features:
        D = preprocess_apply(D, stdSlr_kmeans, pca, options)
    
        # Cast features to visual words:
        visual_words_test = descriptors2words(D, codebook, options.kmeans, descriptors_per_image)
        
    # Scale visual words:
    visual_words_scaled = stdSlr_VW.transform(visual_words_test)
    
    # Compute accuracy:
    accuracy = 100 * clf.score(visual_words_scaled, test_labels)
    
    # Confusion matrix:
    if options.plot_confusion_matrix:
        plot_confusion_matrix(visual_words_scaled, clf, test_labels)
        
    # Compute ROC curve and ROC area for each class
    if options.compute_roc:
        compute_and_save_roc_curve(clf, test_labels, options, visual_words_scaled)

    return accuracy
    
    
##############################################################################
def read_and_extract_visual_words(images_filenames, detector, codebook, options):
    # extract keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    nimages = len(images_filenames)
#    visual_words = np.zeros((nimages,k*21), dtype=np.float32)
    nhistsperlevel = [4**l for l in range(options.depth)]
    nwords = sum(nhistsperlevel) * options.kmeans
    visual_words = np.zeros((nimages, nwords), dtype=np.float32)
    for i in range(nimages):
        filename = images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        visual_words[i,:] = spatial_pyramid_new(gray, detector, codebook, options)

    return visual_words
    
    
##############################################################################
def spatial_pyramid_new(gray, detector, codebook, options):
    
    height, width = gray.shape
    
    visual_words = []
    
    for level in range(options.depth):
        deltai = height / (2**level)
        deltaj = width / (2**level)
        for i in range(2**level):
            for j in range(2**level):
                im = gray[i*deltai : (i+1)*deltai, j*deltaj : (j+1)*deltaj]
                visual_words_im = extract_visual_words(im, detector, codebook, \
                        options.kmeans, options.detector_options)
                visual_words.extend(visual_words_im)
                
    return visual_words
    
    
##############################################################################
#def spatial_pyramids(gray_l2, detector, codebook, k):
#    #Level 2
#    visual_words_l2 = extract_visual_words(gray_l2, detector, codebook, k, detector_options)
#    
#    #Level 1
#    height_l2, width_l2 = gray_l2.shape
#    width_l1 = width_l2 / 2
#    height_l1 = height_l2 / 2
#    
#    gray_l1_1 = gray_l2[0:height_l1, 0:width_l1]
#    visual_words_l1_1 = extract_visual_words(gray_l1_1, detector, codebook, k, detector_options)
#    gray_l1_2 = gray_l2[0:height_l1,width_l1:width_l2]
#    visual_words_l1_2 = extract_visual_words(gray_l1_2, detector, codebook, k, detector_options)
#    gray_l1_3 = gray_l2[height_l1:height_l2,0:width_l1]
#    visual_words_l1_3 = extract_visual_words(gray_l1_3, detector, codebook, k, detector_options)
#    gray_l1_4 = gray_l2[height_l1:height_l2,width_l1:width_l2]
#    visual_words_l1_4 = extract_visual_words(gray_l1_4, detector, codebook, k, detector_options)
#    
#    visual_words_l1 = np.concatenate((visual_words_l1_1, visual_words_l1_2, visual_words_l1_3,visual_words_l1_4),axis=0)
#
#    #Level 0
#    width_l0 = width_l1 / 2
#    heigth_l0 = height_l1 / 2
#    
#    gray_l0_1_1 = gray_l1_1[0:heigth_l0, 0:width_l0]
#    visual_words_l0_1_1 = extract_visual_words(gray_l0_1_1, detector, codebook, k, detector_options)
#    gray_l0_1_2 = gray_l1_1[0:heigth_l0,width_l0:width_l1]
#    visual_words_l0_1_2 = extract_visual_words(gray_l0_1_2, detector, codebook, k, detector_options)
#    gray_l0_1_3 = gray_l1_1[heigth_l0:height_l1,0:width_l0]
#    visual_words_l0_1_3 = extract_visual_words(gray_l0_1_3, detector, codebook, k, detector_options)
#    gray_l0_1_4 = gray_l1_1[heigth_l0:height_l1,width_l0:width_l1]
#    visual_words_l0_1_4 = extract_visual_words(gray_l0_1_4, detector, codebook, k, detector_options)
#    gray_l0_2_1 = gray_l1_2[0:heigth_l0, 0:width_l0]
#    visual_words_l0_2_1 = extract_visual_words(gray_l0_2_1, detector, codebook, k, detector_options)
#    gray_l0_2_2 = gray_l1_2[0:heigth_l0,width_l0:width_l1]
#    visual_words_l0_2_2 = extract_visual_words(gray_l0_2_2, detector, codebook, k, detector_options)
#    gray_l0_2_3 = gray_l1_2[heigth_l0:height_l1,0:width_l0]
#    visual_words_l0_2_3 = extract_visual_words(gray_l0_2_3, detector, codebook, k, detector_options)
#    gray_l0_2_4 = gray_l1_2[heigth_l0:height_l1,width_l0:width_l1]
#    visual_words_l0_2_4 = extract_visual_words(gray_l0_2_4, detector, codebook, k, detector_options)
#    gray_l0_3_1 = gray_l1_3[0:heigth_l0, 0:width_l0]
#    visual_words_l0_3_1 = extract_visual_words(gray_l0_3_1, detector, codebook, k, detector_options)
#    gray_l0_3_2 = gray_l1_3[0:heigth_l0,width_l0:width_l1]
#    visual_words_l0_3_2 = extract_visual_words(gray_l0_3_2, detector, codebook, k, detector_options)
#    gray_l0_3_3 = gray_l1_3[heigth_l0:height_l1,0:width_l0]
#    visual_words_l0_3_3 = extract_visual_words(gray_l0_3_3, detector, codebook, k, detector_options)
#    gray_l0_3_4 = gray_l1_3[heigth_l0:height_l1,width_l0:width_l1]
#    visual_words_l0_3_4 = extract_visual_words(gray_l0_3_4, detector, codebook, k, detector_options)
#    gray_l0_4_1 = gray_l1_4[0:heigth_l0, 0:width_l0]
#    visual_words_l0_4_1 = extract_visual_words(gray_l0_4_1, detector, codebook, k, detector_options)
#    gray_l0_4_2 = gray_l1_4[0:heigth_l0,width_l0:width_l1]
#    visual_words_l0_4_2 = extract_visual_words(gray_l0_4_2, detector, codebook, k, detector_options)
#    gray_l0_4_3 = gray_l1_4[heigth_l0:height_l1,0:width_l0]
#    visual_words_l0_4_3 = extract_visual_words(gray_l0_4_3, detector, codebook, k, detector_options)
#    gray_l0_4_4 = gray_l1_4[heigth_l0:height_l1,width_l0:width_l1]
#    visual_words_l0_4_4 = extract_visual_words(gray_l0_4_4, detector, codebook, k, detector_options)
#    
#    visual_words_l0 = np.concatenate( (\
#        visual_words_l0_1_1, visual_words_l0_1_2, visual_words_l0_1_3,visual_words_l0_1_4, \
#        visual_words_l0_2_1, visual_words_l0_2_2, visual_words_l0_2_3,visual_words_l0_2_4, \
#        visual_words_l0_3_1, visual_words_l0_3_2, visual_words_l0_3_3,visual_words_l0_3_4, \
#        visual_words_l0_4_1, visual_words_l0_4_2, visual_words_l0_4_3,visual_words_l0_4_4),axis=0)
#    
#    visual_words = np.concatenate((1/4 * visual_words_l2, 1/4 * visual_words_l1, 1/2 * visual_words_l0),axis=0)
#            
#    return visual_words
    
    
##############################################################################
def extract_visual_words(gray, detector, codebook, kmeans, detector_options):
    if detector_options.dense_sampling == 1:
        kpt = dense_sampling(detector_options.dense_sampling_max_nr_keypoints, \
                    detector_options.dense_sampling_keypoint_step_size, \
                    detector_options.dense_sampling_keypoint_radius, gray.shape[0], \
                    gray.shape[1])
        kpt, des = detector.compute(gray, kpt)
        #descriptors_per_image[i] = kpt.__len__()
    else:
        kpt, des = detector.detectAndCompute(gray,None)
    
    if not kpt:
        words = []
    else:
        words=codebook.predict(des)
        
    visual_words = np.bincount(words,minlength=kmeans)
    
    return visual_words
    
#############################################################################
def plot_confusion_matrix(visual_words_scaled, clf, test_labels):
    # This function prints and plots the confusion matrix.
    # Normalization can be applied by setting `normalize=True`.

    print 'Computing confusion matrix...'
    sys.stdout.flush()
    
    cmap=plt.cm.Blues
    predictions = clf.predict(visual_words_scaled)
    cm = confusion_matrix(test_labels, predictions)
    plt.figure()
    classes = set(test_labels)
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.show()
    print 'Done!'
    

##############################################################################
def compute_and_save_roc_curve(clf, test_labels, options, visual_words_scaled):
    # Compute ROC curve and ROC area for each class

#    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', 'forest', 'tallbuilding', 'highway']
    classes = set(test_labels)
    
    # Compute probabilities:
    predicted_probabilities = clf.predict_proba(visual_words_scaled)
    # Binarize the labels
    binary_labels = label_binarize(test_labels, classes=classes)
    
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'black', 'red'])

    for i, color in zip(range(classes.__len__()), colors):
        fpr, tpr, thresholds = roc_curve(binary_labels[:, i], predicted_probabilities[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=color,
             label='Label \'%s\' (AUC = %0.2f)' % (classes[i], roc_auc))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right", fontsize='x-small')
    file_name = "roc_curve_desc_%s_dense_%s_kmeans_%s_pyramids_%s_kernel_%s_C_%s_sigma_%s.png" % (options.detector_options.descriptor, \
                                                     options.detector_options.dense_sampling, options.kmeans, options.spatial_pyramids, \
                                                              options.SVM_options.kernel, options.SVM_options.C, options.SVM_options.sigma)
    plt.savefig(file_name, bbox_inches='tight')
    #plt.show()


##############################################################################
class SVM_options_class:
# Options for SVM classifier.
    kernel = 'linear'
    C = 1
    sigma = 1
    degree = 3
    coef0 = 0
    probability = 1 # This changes the way we aggregate predictions. Necessary for ROC.
    

##############################################################################
class detector_options_class:
# Options feature detectors.
    descriptor = 'SIFT'
    nfeatures = 100
    SURF_hessian_ths = 400
    dense_sampling = 0  # Apply dense sampling to the selected detector
    dense_sampling_max_nr_keypoints = 1500  # Maximum number of equally spaced keypoints
    dense_sampling_keypoint_step_size = 10
    dense_sampling_keypoint_radius = 5
    

##############################################################################
class general_options_class:
# General options for the system.
    SVM_options = SVM_options_class()
    detector_options = detector_options_class()
    ncomp_pca = 30 # Number of components for PCA.
    scale_kmeans = 0 # Scale features before applying k-means.
    apply_pca = 0 # Apply, or not, PCA.
    kmeans = 512 # Number of clusters for k-means (codebook).
    k_cv = 5 # Number of subsets for k-fold cross-validation.
    compute_codebook = 1 # Compute or read the codebook.
    fname_codebook = 'codebook512' # In case of reading the codebook, specify here the name of the file.
    spatial_pyramids = 0 # Apply spatial pyramids in BoW framework or not
    depth = 3 # Numbef of levels of the spatial pyramid.
    plot_confusion_matrix = 1 # Compute and show, or not, the confusion matrix.
    compute_roc = 1 # Compute and show ROC.