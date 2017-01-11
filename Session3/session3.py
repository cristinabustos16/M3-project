import cv2
import numpy as np
import cPickle
import time
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import cluster
import sys
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score


##############################################################################
def main(options):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    # Read codebook
    codebook = read_codebook(options.fname_codebook)
    
    # Create the detector object
    detector = create_detector(options.detector_options)
        
    clf, stdSlr_VW = train_system(train_images_filenames, train_labels, detector, \
                                    codebook, options)
    
    accuracy = test_system(test_images_filenames, test_labels, detector, codebook, \
                                    clf, stdSlr_VW, options)
    
    end=time.time()
    running_time = end-start
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Done in '+str(running_time)+' secs.'
    
    return accuracy, running_time


##############################################################################
def train_and_validate(options):
    start = time.time()

    # Compute or read the codebook
    if options.compute_codebook:
        codebook = compute_and_write_codebook(options)
    else:
        codebook = read_codebook(options.fname_codebook)
        
    # Create the cross-validation subsets
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
    
    # Extract the features for all the subsets
    subset_visual_words = list(xrange(options.k_cv))
    # Loop over the subsets:
    for i in range(options.k_cv):
        subset_visual_words[i] = extract_visual_words_all(subsets_filenames[i], detector, codebook, options)
        
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
        # with the train set we have just built, and test with the evaluation
        # set.
        
        # Train system:
        clf, stdSlr_VW = train_system_nocompute(trainset_visual_words, trainset_labels, \
                                            detector, codebook, options)
    
        # Evaluate system:
        accuracy[i] = test_system_nocompute(validation_visual_words, validation_labels, \
                                        detector, codebook, clf, stdSlr_VW, options)
    
    # Compute the mean and the standard deviation of the accuracies found:
    accuracy_mean = np.mean(accuracy)
    print('Mean accuracy: ' + str(accuracy_mean))
    accuracy_sd = np.std(accuracy, ddof = 1)
    print('Std. dev. accuracy: ' + str(accuracy_sd))
    
    end = time.time()
    running_time = end-start
    print 'Done in '+str(running_time)+' secs.'
    
    # Write the results in a text file
    report_name = 'report_' + options.file_name + '.txt'
    fd = open(report_name, 'w')
    try:
        fd.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
        fd.write('\n' + 'Std. dev. accuracy: ' + str(accuracy_sd))
        fd.write('\n' + 'Done in ' + str(end - start) + ' secs.')
    except OSError:
        sys.stdout.write('\n' + 'Mean accuracy: ' + str(accuracy_mean))
    fd.close()
    
    return accuracy_mean, accuracy_sd, running_time


##############################################################################
def create_subsets_cross_validation(k_cv):
    # Create a split for k-fold Cross-Validation.
    # Read the whole training set:
    with open('train_images_filenames.dat', 'rb') as f1:  # b for binary
        images_train = cPickle.load(f1)
    train_images_filenames = np.array(images_train)
    with open('train_labels.dat', 'rb') as f2:  # b for binary
        labels_train = cPickle.load(f2)
    train_labels = np.array(labels_train)
    
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
        with open('subset_'+str(i)+'_filenames.dat', 'wb') as f3:  # b for binary
            cPickle.dump(subset_filenames, f3, cPickle.HIGHEST_PROTOCOL)
        #cPickle.dump(subset_filenames, open('subset_'+str(i)+'_filenames.dat', "wb"))
        with open('subset_'+str(i)+'_labels.dat', 'wb') as f4:  # b for binary
            cPickle.dump(subset_labels, f4, cPickle.HIGHEST_PROTOCOL)
        #cPickle.dump(subset_labels, open('subset_'+str(i)+'_labels.dat', "wb"))
#        np.savetxt('subset_'+str(i)+'_filenames.txt', subset_filenames, fmt='%s')
#        np.savetxt('subset_'+str(i)+'_labels.txt', subset_labels, fmt='%s')
        # Update beginning of indexes:
        ini = fin
        
    
##############################################################################
def compute_and_write_codebook(options):
    
    # read the train and test files
    with open('train_images_filenames.dat', 'rb') as f1:  # b for binary
        train_images_filenames = cPickle.load(f1)

    with open('train_labels.dat', 'rb') as f2:  # b for binary
        train_labels = cPickle.load(f2)

    #train_images_filenames = cPickle.load(open('train_images_filenames.dat','rb'))
    #train_labels = cPickle.load(open('train_labels.dat','rb'))

    print 'Loaded ' + str(len(train_images_filenames)) + \
            ' training images filenames with classes ',set(train_labels)

    # Create the detector object
    detector = create_detector(options.detector_options)
    
    # Extract features from train images:
    D, descriptors_per_image = read_and_extract_features(train_images_filenames, detector, options.detector_options)
    
    # Compute the codebook with the features:
    codebook = compute_codebook(options.kmeans, D)
    
    # Write codebook:
    #cPickle.dump(codebook, open(options.fname_codebook+'.dat', "wb"))
    with open(options.fname_codebook+'.dat', 'wb') as f4:  # b for binary
        cPickle.dump(codebook, f4, cPickle.HIGHEST_PROTOCOL)

    return codebook
    

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
        if step_size < 1:
            step_size = 1
            nr_keypoints = (image_height / step_size) * (image_width / step_size)
            break

    if step_size < 1:
        step_size = 1

    kpt = [cv2.KeyPoint(x, y, radius) for y in range(step_size-1, image_height-step_size, step_size)
                                        for x in range(step_size-1, image_width-step_size, step_size)]

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
        else:
            kpt, des = detector.detectAndCompute(gray,None)
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
def extract_visual_words_all(images_filenames, detector, codebook, options):
    # extract keypoints and descriptors
    # store descriptors in a python list of numpy arrays
    nimages = len(images_filenames)
    
    if options.spatial_pyramids:
        if options.spatial_pyramids_conf == '2x2':
            nhistsperlevel = [4**l for l in range(options.spatial_pyramids_depth)]
        elif options.spatial_pyramids_conf == '1x2':
            nhistsperlevel = [2**l for l in range(options.spatial_pyramids_depth)]
        elif options.spatial_pyramids_conf == '1x3':
            nhistsperlevel = [3**l for l in range(options.spatial_pyramids_depth)]
        else:
            print 'Configuratin of spatial pyramid not recognized.'
            sys.stdout.flush()
            sys.exit()
        nwords = sum(nhistsperlevel) * options.kmeans
    else:
        nwords = options.kmeans
    
    visual_words = np.zeros((nimages, nwords), dtype=np.float32)
    for i in range(nimages):
        filename = images_filenames[i]
        print 'Reading image ' + filename
        ima = cv2.imread(filename)
        gray = cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if options.spatial_pyramids:
            visual_words[i,:] = spatial_pyramid(gray, detector, codebook, options)
        else:
            visual_words[i,:] = extract_visual_words_one(gray, detector, codebook, options.kmeans, options.detector_options)
    return visual_words
    
    
##############################################################################
def extract_visual_words_one(gray, detector, codebook, kmeans, detector_options):
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
    

##############################################################################
def spatial_pyramid(gray, detector, codebook, options):
    
    height, width = gray.shape
    
    visual_words = []
    
    if(options.spatial_pyramids_conf == '2x2'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (2**level)
            deltaj = width / (2**level)
            for i in range(2**level):
                for j in range(2**level):
                    im = gray[i*deltai : (i+1)*deltai, j*deltaj : (j+1)*deltaj]
                    visual_words_im = extract_visual_words_one(im, detector, codebook, \
                            options.kmeans, options.detector_options)
                    visual_words.extend(visual_words_im)
                    
    elif(options.spatial_pyramids_conf == '1x2'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (2**level)
            for i in range(2**level):
                im = gray[i*deltai : (i+1)*deltai, :]
                visual_words_im = extract_visual_words_one(im, detector, codebook, \
                        options.kmeans, options.detector_options)
                visual_words.extend(visual_words_im)
                    
    elif(options.spatial_pyramids_conf == '1x3'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (3**level)
            for i in range(3**level):
                im = gray[i*deltai : (i+1)*deltai, :]
                visual_words_im = extract_visual_words_one(im, detector, codebook, \
                        options.kmeans, options.detector_options)
                visual_words.extend(visual_words_im)
        
    else:
        print 'Configuratin of spatial pyramid not recognized.'
        sys.stdout.flush()
        sys.exit()
                
    return visual_words
    
    
##############################################################################
def histogramIntersection(M, N):
    m_samples , m_features = M.shape
    n_samples , n_features = N.shape
    K_int = np.zeros(shape=(m_samples,n_samples),dtype=np.float)
    #K_int = 0
    for i in range(m_samples):
        for j in range(n_samples):
            K_int[i][j] = np.sum(np.minimum(M[i],N[j]))
            
    return K_int


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
    elif(SVM_options.kernel == 'histogramIntersection'):
        clf = svm.SVC(kernel=histogramIntersection, C = SVM_options.C, coef0 = SVM_options.coef0, \
                random_state = 1, probability = SVM_options.probability).fit(X, L)
    else:
        print 'SVM kernel not recognized!'
    print 'Done!'
    return clf 
    
    
#############################################################################
def compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name):
    cnf_matrix = confusion_matrix(test_labels, predictions)
    plt.figure()
    classes = set(test_labels)
    
    # Prints and plots the confusion matrix.
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    print(cnf_matrix)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if options.save_plots:
        file_name = 'conf_matrix_' + plot_name + '.png'
        plt.savefig(file_name, bbox_inches='tight')
    if options.show_plots:
        plt.show()
        
        
##############################################################################
def compute_and_save_roc_curve(binary_labels, predicted_probabilities, classes, \
                                    options, plot_name):
    # Compute ROC curve and ROC area for each class
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'black', 'red'])
    plt.figure()
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
    if options.save_plots:
        file_name = 'roc_curve_' + plot_name + '.png'
        plt.savefig(file_name, bbox_inches='tight')
    if options.show_plots:
        plt.show()
    
    
##############################################################################
def compute_and_save_precision_recall_curve(binary_labels, predicted_score, \
                                            classes, options, plot_name):
    # Compute Precision-Recall and plot curve
    precision = dict()
    recall = dict()
    average_precision = dict()
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'black', 'red'])

    for i in range(classes.__len__()):
        precision[i], recall[i], _ = precision_recall_curve(binary_labels[:, i], predicted_score[:, i])
        average_precision[i] = average_precision_score(binary_labels[:, i], predicted_score[:, i])

    plt.figure()
    for i, color in zip(range(classes.__len__()), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Label \'%s\' (Avg. precision = %0.2f)'
                       % (classes[i],average_precision[i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right", fontsize='x-small')
    if options.save_plots:
        file_name = 'prec_recall__curve_' + plot_name + '.png'
        plt.savefig(file_name, bbox_inches='tight')
    if options.show_plots:
        plt.show()


##############################################################################
def train_system(train_images_filenames, train_labels, detector, codebook, options):
    # Train the system with the training data.
    
    # Extract the visual words from the train images:
    train_visual_words = extract_visual_words_all(train_images_filenames, \
                                                detector, codebook, options)
    
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(train_visual_words)
    
    # Scale words:
    train_visual_words_scaled = stdSlr_VW.transform(train_visual_words)
    
    # Train the classifier:
    clf = train_classifier(train_visual_words_scaled, train_labels, options.SVM_options)
    
    return clf, stdSlr_VW


##############################################################################
def train_system_nocompute(train_visual_words, train_labels, detector, codebook, options):
    # Train the system with the training data.
    
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(train_visual_words)
    
    # Scale words:
    train_visual_words_scaled = stdSlr_VW.transform(train_visual_words)
    
    # Train the classifier:
    clf = train_classifier(train_visual_words_scaled, train_labels, options.SVM_options)
    
    return clf, stdSlr_VW
    
    
##############################################################################
def test_system(test_images_filenames, test_labels, detector, codebook, clf, \
                                        stdSlr_VW, options):
    # Measure the performance of the system with the test set.    
    
    # Extract the visual words from the test images:
    test_visual_words = extract_visual_words_all(test_images_filenames, detector,\
                                                    codebook, options)
    
    # Scale words:
    test_visual_words_scaled = stdSlr_VW.transform(test_visual_words)
    
    # Compute the accuracy:
    accuracy = 100 * clf.score(test_visual_words_scaled, test_labels)
    
    # Only if pass a valid file descriptor
    if options.compute_evaluation == 1:
        final_issues(test_visual_words_scaled, test_labels, clf, options)

    return accuracy
    
    
##############################################################################
def test_system_nocompute(test_visual_words, test_labels, detector, codebook, clf, \
                                        stdSlr_VW, options):
    # Measure the performance of the system with the test set.    

    # Scale words:
    test_visual_words_scaled = stdSlr_VW.transform(test_visual_words)
    
    # Compute the accuracy:
    accuracy = 100 * clf.score(test_visual_words_scaled, test_labels)
    
    # Only if pass a valid file descriptor
    if options.compute_evaluation == 1:
        final_issues(test_visual_words_scaled, test_labels, clf, options)

    return accuracy
    
    
##############################################################################
def final_issues(test_visual_words_scaled, test_labels, clf, options):
    
    # Get the predictions:
    predictions = clf.predict(test_visual_words_scaled)
    
    plot_name = options.file_name + '_test'
    target_names = ['class mountain', 'class inside_city', 'class Opencountry', \
                    'class coast', 'class street', 'class forest', \
                    'class tallbuilding', 'class highway']
    classes = ['mountain', 'inside_city', 'Opencountry', 'coast', 'street', \
                    'forest', 'tallbuilding', 'highway']
                
    # Report file:
    fid = open('report.txt', 'w')
    fid.write(classification_report(test_labels, predictions, target_names=target_names))
    fid.close()
    
    # Confussion matrix:
    compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name)

    # Compute probabilities:
    predicted_probabilities = clf.predict_proba(test_visual_words_scaled)
    predicted_score = clf.decision_function(test_visual_words_scaled)
    
    # Binarize the labels
    binary_labels = label_binarize(test_labels, classes=classes)
    
    # Compute ROC curve and ROC area for each class
    compute_and_save_roc_curve(binary_labels, predicted_probabilities, classes, \
                                            options, plot_name)
    
    # Compute Precision-Recall curve for each class
    compute_and_save_precision_recall_curve(binary_labels, predicted_score, classes, \
                                            options, plot_name)
        
        
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
    compute_subsets = 1 # Compute or not the cross-validation subsets
    k_cv = 5 # Number of subsets for k-fold cross-validation.
    compute_codebook = 1 # Compute or read the codebook.
    fname_codebook = 'codebook512' # In case of reading the codebook, specify here the name of the file.
    spatial_pyramids = 0 # Apply spatial pyramids in BoW framework or not
    spatial_pyramids_depth = 3 # Number of levels of the spatial pyramid.
    spatial_pyramids_conf = '2x2' # Spatial pyramid configuracion ('2x2', '3x1', '2x1')
    file_name = 'test'
    show_plots = 0
    save_plots = 0
    compute_evaluation = 0
