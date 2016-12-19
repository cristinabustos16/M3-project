import os
import random
import shutil
import cPickle

def create_validation_set(mit_split_images_dir, image_type, final_list_validation_images, final_list_train_images, validation_labels, train_labels):
    #Copy a random 20% of images to the validation folder and delete them from the train folder

    path_train = mit_split_images_dir + '\\train\\' + image_type
    dir = mit_split_images_dir + '\\validation\\'
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = mit_split_images_dir + '\\validation\\' + image_type
    if not os.path.exists(dir):
        os.makedirs(dir)

    files = os.listdir(path_train)
    dir_jpg_files = []
    for file in files:
        if file.endswith(".jpg"):
            dir_jpg_files.append(path_train + '\\' + file)

    dir_val_jpg_files = []
    nr_files = len(dir_jpg_files);
    total_validation_files = int(round((nr_files * 20) / 100))
    for i in range(1, total_validation_files):
        random_file = random.choice(dir_jpg_files)
        dir_jpg_files.remove(random_file)
        assert not os.path.isabs(random_file)
        dstdir = os.path.join(dir, os.path.basename(random_file))
        dir_val_jpg_files.append(dstdir)
        shutil.copy(random_file, dstdir)
        #os.remove(random_file)


    for i in range(len(dir_val_jpg_files)):
        final_list_validation_images.append(dir_val_jpg_files[i])

    for i in range(len(dir_val_jpg_files)):
        validation_labels.append(image_type)

    for i in range(len(dir_jpg_files)):
        final_list_train_images.append(dir_jpg_files[i])

    for i in range(len(dir_jpg_files)):
        train_labels.append(image_type)


mit_split_images_dir = r'..\..\Databases\MIT_split'
validation_images_filenames = []
train_images_selection_filenames = []
validation_labels = []
train_selection_labels = []

create_validation_set(mit_split_images_dir, 'coast', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'forest', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'highway', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'inside_city', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'mountain', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'Opencountry', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'street', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)
create_validation_set(mit_split_images_dir, 'tallbuilding', validation_images_filenames, train_images_selection_filenames, validation_labels, train_selection_labels)

with open('validation_images_filenames.dat', 'wb') as f:
    cPickle.dump(validation_images_filenames, f)

with open('train_images_selection_filenames.dat', 'wb') as f:
    cPickle.dump(train_images_selection_filenames, f)

with open('validation_labels.dat', 'wb') as f:
    cPickle.dump(validation_labels, f)

with open('train_selection_labels.dat', 'wb') as f:
    cPickle.dump(train_selection_labels, f)