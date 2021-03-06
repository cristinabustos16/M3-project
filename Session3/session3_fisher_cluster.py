import cv2
import numpy as np
import cPickle as pickle
import time
import math
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import sys
from sklearn.metrics import confusion_matrix
import itertools
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from yael import ynumpy


##############################################################################
def main(options):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames, \
        test_images_filenames, \
        train_labels, \
        test_labels = read_dataset(options)
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
        
    # Create the detector object
    detector = create_detector(options.detector_options)
        
    # Train system:
    stdSlr_features, pca, gmm, stdSlr, clf = train_system(train_images_filenames, train_labels, detector, options)
    
    # Evaluate system:
    accuracy = test_system(test_images_filenames, test_labels, detector, stdSlr_features, pca, gmm, stdSlr, clf, options)
    
    end=time.time()
    running_time = end-start
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Done in '+str(running_time)+' secs.'
    
    return accuracy, running_time

##############################################################################
def train_and_validate(options):
    start = time.time()
        
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
        with open('subset_'+str(i)+'_filenames.dat', 'rb') as f1:  # b for binary
            subsets_filenames[i] = pickle.load(f1)
        with open('subset_'+str(i)+'_labels.dat', 'rb') as f2:  # b for binary
            subsets_labels[i] = pickle.load(f2)
        
    # Create the detector object
    detector = create_detector(options.detector_options)
    
    # Initialize vector to store the accuracy of each training.
    accuracy = np.zeros(options.k_cv)
    
    # Train and evaluate k times:
    for i in range(options.k_cv):
        # First, we create a list with the names of the files we will use for
        # training. These are the files in all the subsets except the i-th one.
        # The same for the labels.
        # Initialize:
        trainset_images_filenames = []
        trainset_labels = []
        for j in range(options.k_cv):
            if(i != j): # If it is not the i-th subset, add it to our trainset.
                trainset_images_filenames.extend(subsets_filenames[j])
                trainset_labels.extend(subsets_labels[j])
        # For validation, we will use the rest of the images, i.e., the
        # subset i.
        validation_images_filenames = subsets_filenames[i]
        validation_labels = subsets_labels[i]
    
        # The rest is exactly the same as a normal training-testing: we train
        # with the train set we have just built, and test with the evaluation
        # set.
        
        # Train system:
        stdSlr_features, pca, gmm, stdSlr, clf = train_system(trainset_images_filenames, trainset_labels, detector, options)
    
        # Evaluate system:
        accuracy[i] = test_system(validation_images_filenames, validation_labels, detector, stdSlr_features, pca, gmm, stdSlr, clf, options)
    
    # Compute the mean and the standard deviation of the accuracies found:
    accuracy_mean = np.mean(accuracy)
    print('Mean accuracy: ' + str(accuracy_mean))
    accuracy_sd = np.std(accuracy, ddof = 1)
    print('Std. dev. accuracy: ' + str(accuracy_sd))
    
    end = time.time()
    running_time = end-start
    print 'Done in '+str(running_time)+' secs.'
    
    # Write the results in a text file:
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
def read_dataset(options):
    # Read the names of the files.
    # It is possible to select only a part of these files, for faster computation
    # when checking if the code works properly.
    
    # Read the train and test files
    with open('train_images_filenames.dat', 'rb') as f1:  # b for binary
        train_images_filenames = pickle.load(f1)
    with open('test_images_filenames.dat', 'rb') as f2:  # b for binary
        test_images_filenames = pickle.load(f2)
    with open('train_labels.dat', 'rb') as f3:  # b for binary
        train_labels = pickle.load(f3)
    with open('test_labels.dat', 'rb') as f4:  # b for binary
        test_labels = pickle.load(f4)
       
    if options.reduce_dataset:
        # Select randomly only a percentage of the images.        
        
        # Reduce the train set:
        ntrain = len(train_labels)
        nreduced = int(round(options.percent_reduced_dataset * ntrain / 100))
        train_shuffled = np.random.choice(range(ntrain), nreduced, replace = False)
        aux1_list = []
        aux2_list = []
        for i in range(nreduced):
            aux1_list.append(train_images_filenames[train_shuffled[i]])
            aux2_list.append(train_labels[train_shuffled[i]])
        train_images_filenames = aux1_list
        train_labels = aux2_list
        
        # Reduce the test set:
        ntest = len(test_labels)
        nreduced = int(round(options.percent_reduced_dataset * ntest / 100))
        test_shuffled = np.random.choice(range(ntest), nreduced, replace = False)
        aux1_list = []
        aux2_list = []
        for i in range(nreduced):
            aux1_list.append(test_images_filenames[test_shuffled[i]])
            aux2_list.append(test_labels[test_shuffled[i]])
        test_images_filenames = aux1_list
        test_labels = aux2_list
    
    return train_images_filenames, test_images_filenames, train_labels, test_labels


##############################################################################
def create_subsets_cross_validation(k_cv):
    # Create a split for k-fold Cross-Validation.
    # Read the whole training set:
    with open('train_images_filenames.dat', 'rb') as f1:  # b for binary
        images_train = pickle.load(f1)
    train_images_filenames = np.array(images_train)
    with open('train_labels.dat', 'rb') as f2:  # b for binary
        labels_train = pickle.load(f2)
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
            pickle.dump(subset_filenames, f3, pickle.HIGHEST_PROTOCOL)
        #cPickle.dump(subset_filenames, open('subset_'+str(i)+'_filenames.dat', "wb"))
        with open('subset_'+str(i)+'_labels.dat', 'wb') as f4:  # b for binary
            pickle.dump(subset_labels, f4, pickle.HIGHEST_PROTOCOL)
        #cPickle.dump(subset_labels, open('subset_'+str(i)+'_labels.dat', "wb"))
#        np.savetxt('subset_'+str(i)+'_filenames.txt', subset_filenames, fmt='%s')
#        np.savetxt('subset_'+str(i)+'_labels.txt', subset_labels, fmt='%s')
        # Update beginning of indexes:
        ini = fin
        
##############################################################################
def extract_SIFT_features(gray, detector, detector_options):

    if detector_options.dense_sampling == 1:
        kpt = dense_sampling(detector_options.dense_sampling_max_nr_keypoints, detector_options.dense_sampling_keypoint_step_size, \
                             detector_options.dense_sampling_keypoint_radius, gray.shape[0], gray.shape[1])
        kpt, des = detector.compute(gray, kpt)
    else:
        kpt, des = detector.detectAndCompute(gray,None)

    print str(len(kpt))+' extracted keypoints and descriptors'
    return des
    
##############################################################################
def applyNormalization(D,options):
    D_norm = np.zeros(D.shape)
    if options.normalization == 'power':
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                D_norm[i,j] = math.copysign(math.sqrt(math.fabs(D[i,j])),D[i,j])
    elif options.normalization == 'L2':
        for i in range(D.shape[0]):
            aux = 0
            for j in range(D.shape[1]):
                aux = aux + math.pow(D[i,j],2)
            aux = math.sqrt(aux)
            D_norm[i] = D[i] / aux
    else:
        print 'Normalization function not recognized.'
        sys.stdout.flush()
        sys.exit()
    return D_norm
        
##############################################################################
def train_system(train_filenames, train_labels, detector, options):
    # Read the images and extract the SIFT features.
    Train_descriptors = []
    Train_label_per_descriptor = []
    for i in range(len(train_filenames)):
        filename=train_filenames[i]
        print 'Reading image '+filename
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        if options.spatial_pyramids:
            des = spatial_pyramid(gray, detector, options)
        else:
            des = extract_SIFT_features(gray, detector, options.detector_options)
        Train_descriptors.append(des)
        Train_label_per_descriptor.append(train_labels[i])
        
    # Transform everything to numpy arrays
    D=Train_descriptors[0]
    L=np.array([Train_label_per_descriptor[0]]*Train_descriptors[0].shape[0])
    for i in range(1,len(Train_descriptors)):
        D=np.vstack((D,Train_descriptors[i]))
        L=np.hstack((L,np.array([Train_label_per_descriptor[i]]*Train_descriptors[i].shape[0])))

    stdSlr_features = StandardScaler()
    pca = None
    if options.apply_pca:
        stdSlr_features = StandardScaler().fit(D)
        D = stdSlr_features.transform(D)
        pca = PCA(n_components = options.ncomp_pca)
        pca.fit(D)
        D = pca.transform(D)
    
    print 'Computing gmm with '+str(options.kmeans)+' centroids'
    init=time.time()
    gmm = ynumpy.gmm_learn(np.float32(D), options.kmeans)
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    if options.apply_pca:
        num_features = options.ncomp_pca
    else:
        num_features = 128
    init=time.time()
    fisher=np.zeros((len(Train_descriptors),options.kmeans*num_features*2),dtype=np.float32)
    for i in xrange(len(Train_descriptors)):
        if options.apply_pca:
            descriptor = stdSlr_features.transform(Train_descriptors[i])
            descriptor = pca.trasform(descriptor)
        else:
            descriptor = Train_descriptors[i]
        fisher[i,:]= ynumpy.fisher(gmm, descriptor, include = ['mu','sigma'])
    end=time.time()
    print 'Done in '+str(end-init)+' secs.'
    
    if options.apply_normalization:
        fisher = applyNormalization(fisher, options)
    
    # Train a linear SVM classifier
    stdSlr = StandardScaler().fit(fisher)
    D_scaled = stdSlr.transform(fisher)
    print 'Training the SVM classifier...'
    clf = svm.SVC(kernel='linear', C=1).fit(D_scaled, train_labels)
    print 'Done!'
    
    return stdSlr_features, pca, gmm, stdSlr, clf
    
#############################################################################
def test_system(test_filenames, test_labels, detector, stdSlr_features, pca, gmm, stdSlr, clf, options):
    if options.apply_pca:
        num_features = options.ncomp_pca
    else:
        num_features = 128
    fisher_test=np.zeros((len(test_filenames),options.kmeans*num_features*2),dtype=np.float32)
    for i in range(len(test_filenames)):
        filename=test_filenames[i]
        print 'Reading image '+filename
        ima=cv2.imread(filename)
        gray=cv2.cvtColor(ima,cv2.COLOR_BGR2GRAY)
        kpt,des=detector.detectAndCompute(gray,None)
        
        if options.apply_pca:
            des = stdSlr_features.transform(des)
            des = pca.transform(des)
        
        fisher_test[i,:]=ynumpy.fisher(gmm, des, include = ['mu','sigma'])
        
        if options.apply_normalization:
            fisher_test = applyNormalization(fisher_test,options)
        
    test_fisher_vectors_scaled = stdSlr.transform(fisher_test)
    accuracy = 100*clf.score(test_fisher_vectors_scaled, test_labels)
    
    if options.evaluation_measures:
        final_issues(test_fisher_vectors_scaled, test_labels, clf, options)
    
    return accuracy
    
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
def spatial_pyramid(gray, detector, options):
    
    height, width = gray.shape
    
    des = []
    
    if(options.spatial_pyramids_conf == '2x2'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (2**level)
            deltaj = width / (2**level)
            for i in range(2**level):
                for j in range(2**level):
                    im = gray[i*deltai : (i+1)*deltai, j*deltaj : (j+1)*deltaj]
                    des_im = extract_SIFT_features(im, detector, options.detector_options)
                    des.extend(des_im)
                    
    elif(options.spatial_pyramids_conf == '1x2'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (2**level)
            for i in range(2**level):
                im = gray[i*deltai : (i+1)*deltai, :]
                des_im = extract_SIFT_features(im, detector, options.detector_options)
                des.extend(des_im)
                    
    elif(options.spatial_pyramids_conf == '1x3'):
        for level in range(options.spatial_pyramids_depth):
            deltai = height / (3**level)
            for i in range(3**level):
                im = gray[i*deltai : (i+1)*deltai, :]
                des_im = extract_SIFT_features(im, detector, options.detector_options)
                des.extend(des_im)
        
    else:
        print 'Configuratin of spatial pyramid not recognized.'
        sys.stdout.flush()
        sys.exit()
                
    return des
    
    
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
def train_classifier(X, L, options):
# Select the classifier for training.
    if(options.classifier == 'svm'):
        clf = train_SVM(X, L, options.SVM_options)
    elif(options.classifier == 'rf'):
        clf = train_randomforest(X, L, options.rf_options)
    elif(options.classifier == 'adaboost'):
        clf = train_adaboost(X, L, options.adaboost_options)
    else:
        print 'SVM kernel not recognized!'
    return clf


##############################################################################
def train_randomforest(X, L, rf_options):
# Train the Random Forest, given some options and the training data.
    print 'Training the Random Forest classifier...'
    sys.stdout.flush()
    clf = RandomForestClassifier(n_estimators = rf_options.n_estimators, \
                                    max_features = rf_options.max_features, \
                                    max_depth = rf_options.max_depth, \
                                    min_samples_split = rf_options.min_samples_split, \
                                    min_samples_leaf = rf_options.min_samples_leaf).fit(X, L)
    print 'Done!'
    return clf


##############################################################################
def train_adaboost(X, L, adaboost_options):
# Train the Adaboost classifier, given some options and the training data.
    print 'Training the Adaboost classifier...'
    sys.stdout.flush()
    clf = AdaBoostClassifier(base_estimator = adaboost_options.base_estimator, \
                                n_estimators = adaboost_options.n_estimators, \
                                learning_rate = adaboost_options.learning_rate).fit(X, L)
    print 'Done!'
    return clf


##############################################################################
def train_SVM(X, L, SVM_options):
# Train the SVM, given some options and the training data.
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
#def compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name):
#    cnf_matrix = confusion_matrix(test_labels, predictions)
#    plt.figure()
#    classes = set(test_labels)
#    
#    # Prints and plots the confusion matrix.
#    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
#    plt.title('Confusion matrix')
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    print(cnf_matrix)
#
#    thresh = cnf_matrix.max() / 2.
#    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
#        plt.text(j, i, cnf_matrix[i, j],
#                 horizontalalignment="center",
#                 color="white" if cnf_matrix[i, j] > thresh else "black")
#
#    plt.tight_layout()
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    
#    if options.save_plots:
#        file_name = 'conf_matrix_' + plot_name + '.png'
#        plt.savefig(file_name, bbox_inches='tight')
#    if options.show_plots:
#        plt.show()
        
        
##############################################################################
#def compute_and_save_roc_curve(binary_labels, predicted_probabilities, classes, \
#                                    options, plot_name):
#    # Compute ROC curve and ROC area for each class
#    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'black', 'red'])
#    plt.figure()
#    for i, color in zip(range(classes.__len__()), colors):
#        fpr, tpr, thresholds = roc_curve(binary_labels[:, i], predicted_probabilities[:, i])
#        roc_auc = auc(fpr, tpr)
#        plt.plot(fpr, tpr, lw=2, color=color,
#             label='Label \'%s\' (AUC = %0.2f)' % (classes[i], roc_auc))
#
#    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#    plt.xlim([-0.05, 1.05])
#    plt.ylim([-0.05, 1.05])
#    plt.xlabel('False Positive Rate')
#    plt.ylabel('True Positive Rate')
#    plt.title('ROC Curve')
#    plt.legend(loc="lower right", fontsize='x-small')
#    if options.save_plots:
#        file_name = 'roc_curve_' + plot_name + '.png'
#        plt.savefig(file_name, bbox_inches='tight')
#    if options.show_plots:
#        plt.show()
    
    
##############################################################################
#def compute_and_save_precision_recall_curve(binary_labels, predicted_score, \
#                                            classes, options, plot_name):
#    # Compute Precision-Recall and plot curve
#    precision = dict()
#    recall = dict()
#    average_precision = dict()
#    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'black', 'red'])
#
#    for i in range(classes.__len__()):
#        precision[i], recall[i], _ = precision_recall_curve(binary_labels[:, i], predicted_score[:, i])
#        average_precision[i] = average_precision_score(binary_labels[:, i], predicted_score[:, i])
#
#    plt.figure()
#    for i, color in zip(range(classes.__len__()), colors):
#        plt.plot(recall[i], precision[i], color=color, lw=2,
#                 label='Label \'%s\' (Avg. precision = %0.2f)'
#                       % (classes[i],average_precision[i]))
#
#    plt.xlim([0.0, 1.0])
#    plt.ylim([0.0, 1.05])
#    plt.xlabel('Recall')
#    plt.ylabel('Precision')
#    plt.title('Precision-Recall Curve')
#    plt.legend(loc="lower right", fontsize='x-small')
#    if options.save_plots:
#        file_name = 'prec_recall__curve_' + plot_name + '.png'
#        plt.savefig(file_name, bbox_inches='tight')
#    if options.show_plots:
#        plt.show()
    
##############################################################################
def final_issues(test_fisher_vectors_scaled, test_labels, clf, options):
    
    # Get the predictions:
    predictions = clf.predict(test_fisher_vectors_scaled)
    
    plot_name = options.file_name + '_test'

    classes = clf.classes_
                
    # Report file:
    fid = open('report.txt', 'w')
    fid.write(classification_report(test_labels, predictions, target_names=classes))
    fid.close()
    
    # Confussion matrix:
#    compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name)

    # Compute probabilities:
    predicted_probabilities = clf.predict_proba(test_fisher_vectors_scaled)
    # predicted_score = clf.decision_function(test_fisher_vectors_scaled)
    
    # Binarize the labels
    binary_labels = label_binarize(test_labels, classes=clf.classes_)

    # Compute ROC curve and ROC area for each class
#    compute_and_save_roc_curve(binary_labels, predicted_probabilities, clf.classes_, \
#                                            options, plot_name)
    
    # Compute Precision-Recall curve for each class
#    compute_and_save_precision_recall_curve(binary_labels, predicted_probabilities, clf.classes_, \
#                                            options, plot_name)
        
        
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
class random_forest_options_class:
# Options for Random Forest classifier.
    n_estimators = 10
    max_features = 'auto'
    max_depth = None
    min_samples_split = 2
    min_samples_leaf = 1
        
        
##############################################################################
class adaboost_options_class:
# Options for Random Forest classifier.
    base_estimator = DecisionTreeClassifier
    n_estimators = 50
    learning_rate = 1.
    

##############################################################################
class detector_options_class:
# Options feature detectors.
    descriptor = 'SIFT'
    nfeatures = 100
    SURF_hessian_ths = 400
    dense_sampling = 0  # Apply dense sampling to the selected detector
    dense_sampling_max_nr_keypoints = 1500  # Maximum number of equally spaced keypoints
    dense_sampling_keypoint_step_size = 8
    dense_sampling_keypoint_radius = 8
    

##############################################################################
class general_options_class:
# General options for the system.
    SVM_options = SVM_options_class()
    rf_options = random_forest_options_class()
    adaboost_options = adaboost_options_class()
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
    spatial_pyramids_depth = 2 # Number of levels of the spatial pyramid.
    spatial_pyramids_conf = '3x1' # Spatial pyramid configuracion ('2x2', '3x1', '2x1')
    file_name = 'test'
    show_plots = 0
    save_plots = 0
    compute_evaluation = 0 # Compute the ROC, confusion matrix, and write report.
    classifier = 'svm' # Type of classifier ('svm', 'rf' for Random Forest, 'adaboost')
    reduce_dataset = 0 # Consider only a part of the dataset. Useful for fast computation and checking code errors.
    percent_reduced_dataset = 10 # Percentage of the dataset to consider.
    apply_normalization = 1
    normalization = 'power'
    evaluation_measures = 0

#############################################################################

# Cluster main

# Select options:
options = general_options_class()

options.file_name = sys.argv[1]

options.compute_codebook = int(sys.argv[2])
options.kmeans = int(sys.argv[3])
options.compute_subsets = int(sys.argv[4])
options.detector_options.nfeatures = int(sys.argv[5])

options.detector_options.dense_sampling = int(sys.argv[6])
options.detector_options.dense_sampling_keypoint_step_size = int(sys.argv[7])
options.detector_options.dense_sampling_keypoint_radius = int(sys.argv[7])

options.spatial_pyramids = int(sys.argv[8])
options.spatial_pyramids_depth = int(sys.argv[9])
options.spatial_pyramids_conf = sys.argv[10]

# Call the cross-validation program:
accuracy_mean, accuracy_sd, running_time = train_and_validate(options)