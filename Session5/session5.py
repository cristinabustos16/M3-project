from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras import backend as K
from keras.utils.visualize_util import plot
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

trainset = 'small'
train_all_layers = True

if trainset == 'large':
  train_data_dir='../../Databases/MIT/train'
else:
  train_data_dir='../../Databases/MIT/train_small'
val_data_dir='../../Databases/MIT/validation'
test_data_dir='../../Databases/MIT/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=20


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
#x = MaxPooling2D((2,2),strides=(2,2),name='pool')(x)
x = Flatten(name='flat')(x)
x = Dense(4096, activation='relu', name='fc')(x)
x = Dense(8, activation='softmax',name='predictions')(x)


model = Model(input=base_model.input, output=x)
plot(model, to_file='modelVGG16b.png', show_shapes=True, show_layer_names=True)
if train_all_layers == False
    for layer in base_model.layers:
      layer.trainable = False
    
    
model.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])
for layer in model.layers:
    print layer.name, layer.trainable

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

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

history=model.fit_generator(train_generator,
        samples_per_epoch=batch_size*(int(400*1881/1881//batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=807)


result = model.evaluate_generator(test_generator, val_samples=807)
print result


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