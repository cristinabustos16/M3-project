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
from itertools import cycle
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model

sys.path.append('.')
from tools_yael import predict_fishergmm
from tools_yael import compute_codebook_gmm

##############################################################################
def main_cnn_SVM(options):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames, \
        test_images_filenames, \
        train_labels, \
        test_labels = read_dataset(options)
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    cnn, clf, stdSlr, pca = train_system_cnn_SVM(train_images_filenames, train_labels, options)
    
    accuracy = test_system_cnn_SVM(test_images_filenames, test_labels, cnn, stdSlr, pca, clf, options)

    end=time.time()
    running_time = end-start
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Done in '+str(running_time)+' secs.'
    
    return accuracy, running_time

##############################################################################
def main_cnn(options):
    start = time.time()
    
    # Read the train and test files
    train_images_filenames, \
        test_images_filenames, \
        train_labels, \
        test_labels = read_dataset(options)
    
    print 'Loaded '+str(len(train_images_filenames))+' training images filenames with classes ',set(train_labels)
    print 'Loaded '+str(len(test_images_filenames))+' testing images filenames with classes ',set(test_labels)
    
    # Create the CNN:
    detector = create_cnn('block5_conv2')
        
    clf, codebook, stdSlr_VW, stdSlr_features, pca = \
            train_system_cnn(train_images_filenames, train_labels, \
                                detector, options)
    
    accuracy = test_system_cnn(test_images_filenames, test_labels, \
                                detector, codebook, clf, stdSlr_VW, \
                                stdSlr_features, pca, options)
    
    end=time.time()
    running_time = end-start
    print 'Accuracy: ' + str(accuracy) + '%'
    print 'Done in '+str(running_time)+' secs.'
    
    return accuracy, running_time
    
##############################################################################   
def create_cnn(layer):
    # Load VGG model
    base_model = VGG16(weights='imagenet')
    # Crop the network:
    cnn = Model(input=base_model.input, output=base_model.get_layer(layer).output)
    return cnn


##############################################################################
def read_dataset(options):
    # Read the names of the files.
    # It is possible to select only a part of these files, for faster computation
    # when checking if the code works properly.
    
    # Read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))
    
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
    with open('train_images_filenames.dat', 'r') as f1:  # b for binary
        images_train = cPickle.load(f1)
    train_images_filenames = np.array(images_train)
    with open('train_labels.dat', 'r') as f2:  # b for binary
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
#        with open('subset_'+str(i)+'_filenames.dat', 'w') as f3:  # b for binary
#            cPickle.dump(subset_filenames, f3, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(subset_filenames, open('subset_'+str(i)+'_filenames.dat', "wb"))
#        with open('subset_'+str(i)+'_labels.dat', 'w') as f4:  # b for binary
#            cPickle.dump(subset_labels, f4, cPickle.HIGHEST_PROTOCOL)
        cPickle.dump(subset_labels, open('subset_'+str(i)+'_labels.dat', "wb"))
#        np.savetxt('subset_'+str(i)+'_filenames.txt', subset_filenames, fmt='%s')
#        np.savetxt('subset_'+str(i)+'_labels.txt', subset_labels, fmt='%s')
        # Update beginning of indexes:
        ini = fin
    

##############################################################################
def compute_codebook_kmeans(kmeans, D):
    # Clustering (unsupervised classification)
    # Apply kmeans over the features to computer the codebook.
    print 'Computing kmeans with ' + str(kmeans) + ' centroids'
    sys.stdout.flush()
    init = time.time()
    codebook = cluster.MiniBatchKMeans(n_clusters=kmeans, verbose=False, \
            batch_size = kmeans * 20, compute_labels=False, \
            reassignment_ratio=10**-4)
    codebook.fit(D)
    end = time.time()
    print 'Done in ' + str(end-init) + ' secs.'
    return codebook
    

##############################################################################
def compute_codebook(D, options):
    # Clustering (unsupervised classification)
    # It can be either K-means or a GMM for Fisher Vectors.
    if options.use_fisher:
        codebook = compute_codebook_gmm(options.kmeans, D)
    else:
        codebook = compute_codebook_kmeans(options.kmeans, D)
    return codebook
    
    
##############################################################################
def read_codebook(options):
    # Read the codebook.
    # Select the name of the file, depending on the options:
    codebookname = 'codebook' + str(options.kmeans)
    if options.detector_options.dense_sampling == 1:
        codebookname = codebookname + '_dense'
    codebookname = codebookname + '.dat'
    with open(codebookname, "r") as input_file:
        codebook = cPickle.load(input_file)
    return codebook

##############################################################################
def read_and_extract_features_cnn_SVM(images_filenames, cnn):
    # Extract features using a CNN.

    descriptors = []
    nimages = len(images_filenames)
    nfeatures_img = np.zeros(nimages, dtype=np.uint)
    print 'Extracting features with CNN...'
    sys.stdout.flush()
    progress = 0
    for i in range(nimages):
        if(np.float32(i) / nimages * 100 > progress + 10):
            progress = 10 * int(round(np.float32(i) / nimages * 10))
            print str(progress) + '% completed'
            sys.stdout.flush()
            
        # Read and process image:
        img = image.load_img(images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract the features using the CNN:
        des = cnn.predict(x)
        # Append to list with features of all images:
        descriptors.append(des)
        # # Number of features per image (height times with of the convolutional layer):
        nfeatures_img[i] = len(des)
    print '100% completed'
    sys.stdout.flush()
    
    # Transform everything to numpy arrays
    size_features = descriptors[0].shape[1] # Length of each feature (depth of the convolutional layer).
    D = np.zeros((nimages * nfeatures_img[i], size_features), dtype=np.float32)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint += len(descriptors[i])
    return D, nfeatures_img
    
##############################################################################
def read_and_extract_features_cnn(images_filenames, cnn):
    # Extract features using a CNN.

    descriptors = []
    nimages = len(images_filenames)
    nfeatures_img = np.zeros(nimages, dtype=np.uint)
    print 'Extracting features with CNN...'
    sys.stdout.flush()
    progress = 0
    for i in range(nimages):
        if(np.float32(i) / nimages * 100 > progress + 10):
            progress = 10 * int(round(np.float32(i) / nimages * 10))
            print str(progress) + '% completed'
            sys.stdout.flush()
            
        # Read and process image:
        img = image.load_img(images_filenames[i], target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Extract the features using the CNN:
        des = cnn.predict(x)
        # Convert to a two-dimensional array:
        des = np.reshape(des, (des.shape[1] * des.shape[2], des.shape[3]))
        # Append to list with features of all images:
        descriptors.append(des)
        # # Number of features per image (height times with of the convolutional layer):
        nfeatures_img[i] = des.shape[0]
    print '100% completed'
    sys.stdout.flush()
    
    # Transform everything to numpy arrays
    size_features = descriptors[0].shape[1] # Length of each feature (depth of the convolutional layer).
    D = np.zeros((nimages * nfeatures_img[i], size_features), dtype=np.float32)
    startingpoint = 0
    for i in range(len(descriptors)):
        D[startingpoint:startingpoint+len(descriptors[i])]=descriptors[i]
        startingpoint += len(descriptors[i])
    
    return D, nfeatures_img
    

##############################################################################
def features2words_all(D, codebook, options, nimages, nfeatures_img):
    # Extract visual words (or Fisher vectors) from features. 
    
    # Calculate the length of the visual words, depending on the
    # configuration (spatial pyramids and Fisher vectors):
    if options.spatial_pyramids:
        if options.spatial_pyramids_conf == '2x2':
            nhistsperlevel = [4**l for l in range(options.spatial_pyramids_depth)]
        elif options.spatial_pyramids_conf == '2x1':
            nhistsperlevel = [2**l for l in range(options.spatial_pyramids_depth)]
        elif options.spatial_pyramids_conf == '3x1':
            nhistsperlevel = [3**l for l in range(options.spatial_pyramids_depth)]
        else:
            print 'Configuratin of spatial pyramid not recognized.'
            sys.stdout.flush()
            sys.exit()
        length_words = sum(nhistsperlevel) * options.kmeans
    else:
        length_words = options.kmeans
    if(options.use_fisher):
        if(options.apply_pca):
            length_words = length_words * options.ncomp_pca * 2
        else:
            length_words = length_words * 128 * 2
    
    visual_words = np.zeros((nimages, length_words), dtype=np.float32)
    
    idx_fin = 0
    for i in range(nimages):
        idx_ini = idx_fin
        idx_fin = idx_ini + nfeatures_img[i]
        # From features to words:
        if(options.use_fisher):
            visual_words[i,:] = predict_fishergmm(codebook, \
                                    D[idx_ini:idx_fin,:], options)
        else:
            visual_words_img = codebook.predict(D[idx_ini:idx_fin,:])
            visual_words[i,:] = np.bincount(visual_words_img, \
                                    minlength=options.kmeans)

    return visual_words


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
    
    print X.__class__.__name__
    print X.shape
    print L.__class__.__name__
    print len(L)
    
    sys.stdout.flush()
    if(SVM_options.kernel == 'linear'):
        clf = svm.SVC(kernel='linear', C = SVM_options.C, \
                probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'poly'):
        clf = svm.SVC(kernel='poly', C = SVM_options.C, degree = SVM_options.degree, \
                coef0 = SVM_options.coef0, \
                probability = SVM_options.probability).fit(X,L)
    elif(SVM_options.kernel == 'rbf'):
        clf = svm.SVC(kernel='rbf', C = SVM_options.C, gamma = SVM_options.sigma, \
                probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'sigmoid'):
        clf = svm.SVC(kernel='sigmoid', C = SVM_options.C, coef0 = SVM_options.coef0, \
                probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'histogramIntersection'):
        clf = svm.SVC(kernel=histogramIntersection, C = SVM_options.C, coef0 = SVM_options.coef0, \
                probability = SVM_options.probability).fit(X, L)
    elif(SVM_options.kernel == 'precomputed'):
        kernelMatrix = histogramIntersection(X,X);
        clf = svm.SVC(kernel=histogramIntersection, C = SVM_options.C, coef0 = SVM_options.coef0, \
                probability = SVM_options.probability).fit(kernelMatrix, L)
    else:
        print 'SVM kernel not recognized!'
    print 'Done!'
    return clf
        
    
##############################################################################
def histogramIntersection(M, N):
    m_samples , m_features = M.shape
    n_samples , n_features = N.shape
    K_int = np.zeros(shape=(m_samples,n_samples),dtype=np.float)
    #K_int = 0
#    for p in range(m_samples):
#        nonzero_ind = [i for (i, val) in enumerate(M[p]) if val > 0]
#        temp_M =  [M[p][index] for index in nonzero_ind]
#        for q in range (n_samples):
#            temp_N =  [N[q][index] for index in nonzero_ind]
#            K_int[p][q] = np.sum(np.minimum(temp_M,temp_N)) 
    for i in range(m_samples):
         for j in range(n_samples):
             K_int[i][j] = np.sum(np.minimum(M[i],N[j]))
    return K_int
    
    
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
        fpr, tpr, thresholds = roc_curve(binary_labels[:,i], predicted_probabilities[:,i])
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
def preprocess_train(D, options):
    # Fit the scaler and the PCA with the training features.
    # Also, give back the features already preprocessed.
    stdSlr_features = StandardScaler()
    if(options.scale_features == 1):
        stdSlr_features = StandardScaler().fit(D)
        D = stdSlr_features.transform(D)
    pca = PCA(n_components = options.ncomp_pca)
    if(options.apply_pca == 1):
        print "Fitting principal components analysis..."
        pca.fit(D)
        D = pca.transform(D)
        print "Explained variance with ", options.ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'
    return D, stdSlr_features, pca
    
    
##############################################################################
def preprocess_fit(D, options):
    # Fit the scaler and the PCA with the training features.
    stdSlr_features = StandardScaler()
    if(options.scale_features == 1):
        stdSlr_features = StandardScaler().fit(D)
        D_scaled = stdSlr_features.transform(D)
    else:
        D_scaled = D
    pca = PCA(n_components = options.ncomp_pca)
    if(options.apply_pca == 1):
        print "Fitting principal components analysis..."
        pca.fit(D_scaled)
        print "Explained variance with ", options.ncomp_pca , \
            " components: ", sum(pca.explained_variance_ratio_) * 100, '%'
    return stdSlr_features, pca
    
    
##############################################################################
def preprocess_apply(D, stdSlr_features, pca, options):
    # Scale and apply PCA to features.
    if(options.scale_features == 1):
        D = stdSlr_features.transform(D)
    if(options.apply_pca == 1):
        D = pca.transform(D)
    return D


##############################################################################
def train_system_cnn(images_filenames, labels, cnn, options):
    # Train the system with the training data.
                        
    nimages = len(images_filenames)

    # Extract features from all images:
    D, nfeatures_img = read_and_extract_features_cnn(images_filenames, cnn)
    
    # Scale and apply PCA:
    D, stdSlr_features, pca = preprocess_train(D, options)
                           
    # Compute or read the codebook:
    if options.compute_codebook:
        codebook = compute_codebook(D, options)
    else:
        codebook = read_codebook(options) 
    
    # Extract visual words or Fisher vectors:
    visual_words = features2words_all(D, codebook, options, nimages, nfeatures_img)
    
    # Fit scaler for words:
    stdSlr_VW = StandardScaler().fit(visual_words)
    # Scale words:
    if(options.SVM_options.kernel != 'histogramIntersection'):
        visual_words = stdSlr_VW.transform(visual_words)
    
    # Train the classifier:
    clf = train_classifier(visual_words, labels, options)
    
    return clf, codebook, stdSlr_VW, stdSlr_features, pca
 
##############################################################################
def train_system_cnn_SVM(images_filenames, labels, options):
    nimages = len(images_filenames)
    
    # create the cnn
    cnn = create_cnn('fc2')
        
    # read the CNN features from last FC layer
    D, nfeatures_img = read_and_extract_features_cnn_SVM(images_filenames, cnn)

    # Scale input data, and keep the scaler for later:
    stdSlr = StandardScaler().fit(D)
    D = stdSlr.transform(D)

    # PCA:
    pca = PCA(n_components = options.ncomp_pca)
    #print "Applying principal components analysis..."
    #pca.fit(D)
    #D = pca.transform(D)

    # Train a linear SVM classifier
    clf = train_classifier(D, labels, options)
    
    return cnn, clf, stdSlr, pca


##############################################################################
def test_system_cnn(images_filenames, labels, cnn, codebook, clf, \
                        stdSlr_VW, stdSlr_features, pca, options):
    # Measure the performance of the system with the test set. 
    nimages = len(images_filenames)

    # Extract features from all images:
    #D, nfeatures_img = read_and_extract_features_cnn_SVM(images_filenames, cnn)
    D, nfeatures_img = read_and_extract_features_cnn(images_filenames, cnn)
    
    # Scale and apply PCA:
    D = preprocess_apply(D, stdSlr_features, pca, options)
    
    # Extract visual words or Fisher vectors:
    visual_words = features2words_all(D, codebook, options, nimages, nfeatures_img)
    
    # Scale words:
    if(options.SVM_options.kernel != 'histogramIntersection'):
        visual_words = stdSlr_VW.transform(visual_words)
    
    # Compute the accuracy:
    accuracy = 100 * clf.score(visual_words, labels)
    
    # Only if pass a valid file descriptor:
    if options.compute_evaluation == 1:
        final_issues(visual_words, labels, clf, options)
    
    return accuracy

#############################################################################
def test_system_cnn_SVM(images_filenames, labels, cnn, stdSlr, pca, clf, options):
    nimages = len(images_filenames)
    
    # Extract features from all images:
    D, nfeatures_img = read_and_extract_features_cnn_SVM(images_filenames, cnn)
    
    # Scale input data, and keep the scaler for later:
    D = stdSlr.transform(D)
    
    # Scale and apply PCA:
    #D = pca.transform(D)

    # Compute the accuracy:
    accuracy = 100 * clf.score(D,labels)
    
    # Only if pass a valid file descriptor:
    if options.compute_evaluation == 1:
        final_issues(descriptors, labels, clf, options)
    
    return accuracy
    
    
##############################################################################
def final_issues(test_visual_words_scaled, test_labels, clf, options):
    
    # Get the predictions:
    predictions = clf.predict(test_visual_words_scaled)
    
    plot_name = options.file_name + '_test'

    classes = clf.classes_
                
    # Report file:
    fid = open('report.txt', 'w')
    fid.write(classification_report(test_labels, predictions, target_names=classes))
    fid.close()
    
    # Confussion matrix:
    compute_and_save_confusion_matrix(test_labels, predictions, options, plot_name)

    # Compute probabilities:
    predicted_probabilities = clf.predict_proba(test_visual_words_scaled)
    # predicted_score = clf.decision_function(test_visual_words_scaled)
    
    # Binarize the labels
    binary_labels = label_binarize(test_labels, classes=clf.classes_)

    # Compute ROC curve and ROC area for each class
    compute_and_save_roc_curve(binary_labels, predicted_probabilities, clf.classes_, \
                                            options, plot_name)
    
    # Compute Precision-Recall curve for each class
    compute_and_save_precision_recall_curve(binary_labels, predicted_probabilities, clf.classes_, \
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
    dense_sampling_keypoint_step_size = 10
    dense_sampling_keypoint_radius = 5
    

##############################################################################
class general_options_class:
# General options for the system.
    SVM_options = SVM_options_class()
    rf_options = random_forest_options_class()
    adaboost_options = adaboost_options_class()
    detector_options = detector_options_class()
    ncomp_pca = 30 # Number of components for PCA.
    scale_features = 0 # Scale features before applying k-means.
    apply_pca = 0 # Apply, or not, PCA.
    kmeans = 512 # Number of clusters for k-means (codebook).
    compute_subsets = 1 # Compute or not the cross-validation subsets
    k_cv = 5 # Number of subsets for k-fold cross-validation.
    compute_codebook = 1 # Compute or read the codebook.
    spatial_pyramids = 0 # Apply spatial pyramids in BoW framework or not
    spatial_pyramids_depth = 3 # Number of levels of the spatial pyramid.
    spatial_pyramids_conf = '2x2' # Spatial pyramid configuracion ('2x2', '3x1', '2x1')
    spatial_pyramid_kernel = 0
    file_name = 'test'
    show_plots = 0
    save_plots = 0
    compute_evaluation = 0 # Compute the ROC, confusion matrix, and write report.
    classifier = 'svm' # Type of classifier ('svm', 'rf' for Random Forest, 'adaboost')
    reduce_dataset = 0 # Consider only a part of the dataset. Useful for fast computation and checking code errors.
    percent_reduced_dataset = 10 # Percentage of the dataset to consider.
    fast_cross_validation = 0 # Use fast or slow cross-validation. The second one allows for more things.
    use_fisher = 0 # Use fisher vectors.
    features_from_cnn = 1 # Use a Convolutional Neural Network to extract the visual features.
    system = 'SVM' #Select the system to apply (SVM, BoW, FV)
