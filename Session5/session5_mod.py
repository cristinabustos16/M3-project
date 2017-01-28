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
from sklearn.preprocessing import label_binarize

# train_data_dir='../../Databases/MIT/train'
# val_data_dir='../../Databases/MIT/validation'
# test_data_dir='../../Databases/MIT/test'
#
# img_width = 224
# img_height=224

cut_on_layer = 'block5_conv2'
add_maxpool = 1
add_dropout = 1

# Select options:
options = general_options_class()
#options.number_of_epoch = 20
options.number_of_epoch = 2
options.batch_size = 16
#options.val_samples = options.batch_size*(int(400/options.batch_size))
options.val_samples = options.batch_size*(int(200/options.batch_size))
# options.test_samples = options.batch_size*(int(400/options.batch_size))
options.model = 'dropout'
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
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

# Build new model:
x = base_model.get_layer(cut_on_layer).output
if(add_maxpool):
    x = MaxPooling2D((2,2),strides=(2,2),name='pool')(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc')(x)
if(add_dropout):
    x = Dropout(options.drop_prob_fc, name='FC Dropout')(x)
x = Dense(8, activation='softmax',name='predictions')(x)

model = Model(input=base_model.input, output=x)
plot(model, to_file='newmodel.png', show_shapes=True, show_layer_names=True)
for layer in base_model.layers:
     layer.trainable = False
    
model.compile(loss='categorical_crossentropy',optimizer= options.optimizer, metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

#preprocessing_function=preprocess_input,
datagen = ImageDataGenerator(featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    # preprocessing_function=preprocess_input,
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

train_generator = datagen.flow_from_directory(options.train_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(options.test_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(options.val_data_dir,
        target_size=(options.img_width, options.img_height),
        batch_size=options.batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch=options.batch_size*(int(200/options.batch_size)+1),
        nb_epoch=options.number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=options.val_samples)


result = model.evaluate_generator(test_generator, val_samples=options.val_samples)
print result

predictions = model.predict_generator(generator = test_generator, \
                            val_samples = options.val_samples)
y_classes = probas_to_classes(predictions)

print y_classes.__class__.__name__
print predictions.__class__.__name__

print set(y_classes)
print y_classes.shape
print y_classes[3]
print y_classes[5]
print y_classes[25]
print predictions.shape
print predictions[0,:]
print predictions[10,:]
print predictions[29,:]

#compute_and_save_confusion_matrix(y_classes, predictions, options, 'prueba_2')

# classes = test_generator.classes

# y_proba = model.predict(test_generator)
# y_classes = probas_to_classes(y_proba)
# compute_and_save_confusion_matrix(y_classes, y_proba, options, 'prueba_1')

# Binarize the labels
# binary_labels = label_binarize(test_generator, classes=test_generator.classes)

# list all data in history

if True:
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