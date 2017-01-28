#### M3 - Session 5 - Task 5
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import numpy as np
import cPickle
import os
import sys


#############################################################################
def train_and_evaluate(optimizer, nepochs, batch_size):

    # Directories:
#    train_data_dir='./minitrain'
#    val_data_dir='./minival'
    
    train_data_dir='/share/mastergpu/MIT/train'
    val_data_dir='/share/mastergpu/MIT/validation'
    
#    test_data_dir='/share/mastergpu/MIT/test'

    # Images size:
    img_width = 224
    img_height=224
    
    # Load VGG model
    base_model = VGG16(weights='imagenet')
    
    # New model
    x = base_model.get_layer('block5_pool').output
    x = Flatten(name='aplanado')(x)
    x = Dense(1000, activation='relu',name='oculta')(x)
    x = Dense(8, activation='softmax',name='predictions')(x)
    newmodel = Model(input=base_model.input, output=x)
    plot(newmodel, to_file='newmodel.png', show_shapes=True, show_layer_names=True)
    
    # Set to not trainable the first layers:
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile
    newmodel.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
    
    # Show which layers are trainable:
#    for layer in newmodel.layers:
#        print layer.name, layer.trainable
    
    #preprocessing_function=preprocess_input,
    datagen = ImageDataGenerator(featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        rotation_range=0.,
        width_shift_range=0.,
        height_shift_range=0.,
        shear_range=0.,
        zoom_range=0.,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None)
    
    train_generator = datagen.flow_from_directory(train_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    
#    test_generator = datagen.flow_from_directory(test_data_dir,
#            target_size=(img_width, img_height),
#            batch_size=batch_size,
#            class_mode='categorical')
    
    validation_generator = datagen.flow_from_directory(val_data_dir,
            target_size=(img_width, img_height),
            batch_size=batch_size,
            class_mode='categorical')
    
#    newmodel.fit_generator(train_generator,
#            samples_per_epoch=188,
#            nb_epoch=nepochs,
#            validation_data=validation_generator,
#            nb_val_samples=80)
    
    newmodel.fit_generator(train_generator,
            samples_per_epoch=batch_size*(int(400*1881/1881//batch_size)+1),
            nb_epoch=nepochs,
            validation_data=validation_generator,
            nb_val_samples=807)
    
    
#    result = newmodel.evaluate_generator(test_generator, val_samples=807)
    result = newmodel.evaluate_generator(validation_generator, val_samples=807)
    accuracy = result[1]
#    print newmodel.metrics_names
#    print result
    
    return accuracy


#############################################################################
def check_case(batch_size, nepochs, optimizer_name, learn_rate, momentum, cases_done):
    # Check if the current case has been already done.
    select_params = 0
    for i in range(len(cases_done)):
        if cases_done[i][0] == batch_size and \
                cases_done[i][1] == nepochs and \
                cases_done[i][2] == optimizer_name and \
                cases_done[i][3] == learn_rate and \
                cases_done[i][4] == momentum:
            select_params = 1
            break
    return select_params


#############################################################################

# Number of trials:
ntrials = 20

# Directory for saving results:
dirResults = './random_search/'
if not os.path.exists(dirResults):
    os.makedirs(dirResults)
    
# Load previous results:
cases_done = [] # Initialize list with done cases.
# List previous files:
previous_files = os.listdir(dirResults)
nprevious = len(previous_files)    
print str(nprevious) + ' previous cases found.'
if nprevious > 0:
    # Get the data from the previous cases:
    for filename in previous_files:
        case = cPickle.load(open(dirResults + filename, 'r'))
        parameters = [] # List with parameters of a given case.
        for i in range(len(case)-1):
            parameters.append(case[i][1])
        cases_done.append(parameters)

# Parameters to search:
batch_size_vec = [10, 20, 40, 60, 80, 100]
#nepochs_vec = [10, 50, 100]
nepochs_vec = [10, 20, 30]
optimizer_name_vec = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
learn_rate_vec = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
momentum_vec = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

#batch_size_vec = [10]
#nepochs_vec = [2]
#optimizer_name_vec = ['SGD']
#learn_rate_vec = [0.0001]
#momentum_vec = [0.0]

# Fixed parameter:
decay = 0

# Do all the trals, sequentially:
for i in range(ntrials):
    # Select parameters:
    select_params = 1
    while select_params:
        print 'Choosing random parameters...'
        batch_size = np.random.choice(batch_size_vec)
        nepochs = np.random.choice(nepochs_vec)
        optimizer_name = np.random.choice(optimizer_name_vec)
        learn_rate = np.random.choice(learn_rate_vec)
        momentum = np.random.choice(momentum_vec)
        # Check this case has not been done yet:
        select_params = check_case(batch_size, nepochs, optimizer_name, learn_rate, momentum, cases_done)
    print 'Done!'
    # Update previous cases:
    parameters = [batch_size, nepochs, optimizer_name, learn_rate, momentum]
    cases_done.append(parameters)
    
    # Prepare the optimizer:
    if optimizer_name == 'SGD':
        optimizer = SGD(lr = learn_rate, decay=0, momentum = momentum)
    elif optimizer_name == 'RMSprop':
        optimizer = RMSprop(lr = learn_rate, rho=0.9, epsilon=1e-08, decay=0)
    elif optimizer_name == 'Adagrad':
        optimizer = Adagrad(lr = learn_rate, epsilon=1e-08, decay=0)
    elif optimizer_name == 'Adadelta':
        optimizer = Adadelta(lr = learn_rate, rho=0.95, epsilon=1e-08, decay=0)
    elif optimizer_name == 'Adam':
        optimizer = Adam(lr = learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    elif optimizer_name == 'Adamax':
        optimizer = Adamax(lr = learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0)
    elif optimizer_name == 'Nadam':
        optimizer = Nadam(lr = learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)
    
    # Train and evaluate the model:    
    accuracy = train_and_evaluate(optimizer, nepochs, batch_size)
    
    # Save results:
    filename = dirResults + 'case_' + str(i+nprevious) + '.dat'
    if os.path.isfile(filename):
        print 'Error: ' + filename + ' already exists.'
        print 'Aborting execution...'
        sys.stdout.flush()
        sys.exit()
    case = []
    case.append(['batch_size', batch_size])
    case.append(['nepochs', nepochs])
    case.append(['optimizer_name', optimizer_name])
    case.append(['learn_rate', learn_rate])
    case.append(['momentum', momentum])
    case.append(['accuracy', accuracy])
    cPickle.dump(case, open(filename, "w"))









