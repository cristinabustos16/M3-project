from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten, Dropout
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from global_functions import general_options_class
from global_functions import compute_and_save_confusion_matrix
from keras.utils.np_utils import probas_to_classes
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import label_binarize

trainset = 'small'
train_all_layers = False

# Select options:
options = general_options_class()
if trainset == 'large':
  options.train_data_dir='../../Databases/MIT/train'
else:
  options.train_data_dir='../../Databases/MIT/train_small'

options.number_of_epoch = 2
options.batch_size = 32
# options.val_samples = options.batch_size*(int(200/options.batch_size)+1)
options.val_samples = 807
options.dropout_enabled = True
options.drop_prob_fc = 0.5


def preprocess_input(x, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        # 'RGB'->'BGR'
        x = x[ ::-1, :, :]
        # Zero-center by mean pixel
        x[ 0, :, :] -= 103.939
        x[ 1, :, :] -= 116.779
        x[ 2, :, :] -= 123.68
    else:
        # 'RGB'->'BGR'
        x = x[:, :, ::-1]
        # Zero-center by mean pixel
        x[:, :, 0] -= 103.939
        x[:, :, 1] -= 116.779
        x[:, :, 2] -= 123.68
    return x
    
# create the base pre-trained model
base_model = VGG16(weights='imagenet')
plot(base_model, to_file='modelVGG16a.png', show_shapes=True, show_layer_names=True)

x = base_model.get_layer('block5_pool').output
x = Convolution2D(512, 3, 3, activation='relu', border_mode='valid', name='block6_conv1')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='valid', name='block6_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='valid', name='block6_conv3')(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc')(x)
if options.dropout_enabled:
    x = Dropout(options.drop_prob_fc, name='FC Dropout')(x)
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(input=base_model.input, output=x)
plot(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)
if train_all_layers == False:
    for layer in base_model.layers:
      layer.trainable = False
    
model.compile(loss='categorical_crossentropy',optimizer= options.optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

#preprocessing_function=preprocess_input,
datagen_train = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function=preprocess_input,
    rotation_range=0.,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.,
    zoom_range=0.1,
    zca_whitening=False,
    channel_shift_range=0.,
    fill_mode='nearest',
    cval=0.,
    horizontal_flip=True,
    vertical_flip=False,
    rescale=0.3)
    
    
datagen_validation = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    preprocessing_function=preprocess_input,
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

train_generator = datagen_train.flow_from_directory(options.train_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')
#        save_to_dir= 'aumented_data/',
#        save_prefix= 'augmented',
#        save_format='.png')

test_generator = datagen_validation.flow_from_directory(options.test_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')

validation_generator = datagen_validation.flow_from_directory(options.val_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch=options.batch_size*(int(400/options.batch_size)+1),
        nb_epoch=options.number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=options.val_samples)


result = model.evaluate_generator(test_generator, val_samples=options.val_samples)
print result[1]

probabilities = model.predict_generator(test_generator, options.val_samples)
predictions = probas_to_classes(probabilities)
y_classes = test_generator.classes
compute_and_save_confusion_matrix(y_classes, predictions, options, 'prueba_2')

# list all data in history

if False:
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig('loss.jpg')
