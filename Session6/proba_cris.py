from keras.models import Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.noise import GaussianNoise
import matplotlib.pyplot as plt
from keras import backend as K

train_data_dir='../../Databases/MIT/train'
val_data_dir='../../Databases/MIT/validation'
test_data_dir='../../Databases/MIT/test'
img_width =128
img_height=128
batch_size=32
number_of_epoch=50
val_samples=200

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

datagen = ImageDataGenerator(featurewise_center=True,
    preprocessing_function=preprocess_input,
    samplewise_center=False,
    featurewise_std_normalization=True,
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

validation_generator = datagen.flow_from_directory(val_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

#Create the model
inputs = Input(batch_shape=(None,img_width,img_height,3))
#x = Convolution2D(8, 5, 5, activation='relu', border_mode='valid', subsample=(3,3), name='conv1')(inputs)
x = Convolution2D(16, 5, 5, activation='relu', border_mode='valid', subsample=(3,3), name='conv2')(inputs)
x = AveragePooling2D((3, 3), strides=(3, 3), name='pool1')(x)
#x = MaxPooling2D((3, 3), strides=(3, 3), name='pool1')(x)
x = Convolution2D(32, 5, 5, activation='relu', border_mode='valid', subsample=(3,3), name='conv3')(x)
x = AveragePooling2D((2, 2), strides=(2, 2), name='pool2')(x)
#x = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(x)
x = BatchNormalization(mode=2)(x)
x = Dropout(0.5, name='FC Dropout')(x)
x = Flatten(name='flatten')(x)
x = Dense(100, activation='relu', name='fc1')(x)
x = Dense(100, activation='relu', name='fc2')(x)
x = GaussianNoise(0.01)(x)
x = Dense(8, activation='softmax',name='predictions')(x)
model = Model(inputs, x, name='example')


#Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

#Fit the model
history=model.fit_generator(train_generator,
        samples_per_epoch=batch_size*(int(1706/batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=val_samples)
        
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

#Evaluate the model
result = model.evaluate_generator(test_generator, val_samples=val_samples)
print('Test accuracy: ' + str(result[1]*100) + '%')
print('Test loss: ' + str(result[0]))