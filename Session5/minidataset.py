import numpy as np
import cPickle
import os
from shutil import copyfile

##############################################################################
def read_dataset(reduce_dataset, percent_reduced_dataset):
    # Read the names of the files.
    # It is possible to select only a part of these files, for faster computation
    # when checking if the code works properly.
    
    # Read the train and test files
    train_images_filenames = cPickle.load(open('train_images_filenames.dat','r'))
    test_images_filenames = cPickle.load(open('test_images_filenames.dat','r'))
    train_labels = cPickle.load(open('train_labels.dat','r'))
    test_labels = cPickle.load(open('test_labels.dat','r'))
    
    if reduce_dataset:
        # Select randomly only a percentage of the images.        
        
        # Reduce the train set:
        ntrain = len(train_labels)
        nreduced = int(round(percent_reduced_dataset * ntrain / 100))
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
        nreduced = int(round(percent_reduced_dataset * ntest / 100))
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

reduce_dataset = 1
percent_reduced_dataset = 10

dir_minitrain = '../../minitrain/'
dir_minival = '../../minival/'

train_images_filenames, test_images_filenames, train_labels, test_labels\
         = read_dataset(reduce_dataset, percent_reduced_dataset)

# Create directories:
all_labels = train_labels + test_labels
for label in set(all_labels):
    if not os.path.exists(dir_minitrain + label):
        os.makedirs(dir_minitrain + label)
    if not os.path.exists(dir_minival + label):
        os.makedirs(dir_minival + label)

# Copy train set:
for i in range(len(train_labels)):
    split_path = train_images_filenames[i].split('/')
    filename = split_path[len(split_path)-1]
    src = train_images_filenames[i]
    dst = dir_minitrain + train_labels[i] + '/' + filename + '.png'
    copyfile(src, dst)

# Copy validation set:
for i in range(len(test_labels)):
    split_path = test_images_filenames[i].split('/')
    filename = split_path[len(split_path)-1]
    src = test_images_filenames[i]
    dst = dir_minival + test_labels[i] + '/' + filename + '.png'
    copyfile(src, dst)
         
         
         
         
         

