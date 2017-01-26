####### pruebas (xian)

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.layers import Dense
from keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

train_data_dir='/share/mastergpu/MIT/train'
val_data_dir='/share/mastergpu/MIT/validation'
test_data_dir='/share/mastergpu/MIT/test'
img_width = 224
img_height=224
batch_size=32
number_of_epoch=10

#load VGG model
base_model = VGG16(weights='imagenet')
#visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

# new model
x = base_model.get_layer('block5_pool').output
x = Flatten(name='aplanado')(x)
x = Dense(1000, activation='relu',name='oculta')(x)
x = Dense(8, activation='softmax',name='predictions')(x)
newmodel = Model(input=base_model.input, output=x)
plot(newmodel, to_file='newmodel.png', show_shapes=True, show_layer_names=True)

# Set to not trainable the first layers:
for layer in base_model.layers:
    layer.trainable = False

# compile
newmodel.compile(loss='categorical_crossentropy',optimizer='adadelta', metrics=['accuracy'])

# Show which layers are trainable:
for layer in newmodel.layers:
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

history=newmodel.fit_generator(train_generator,
        samples_per_epoch=batch_size*(int(400*1881/1881//batch_size)+1),
        nb_epoch=number_of_epoch,
        validation_data=validation_generator,
        nb_val_samples=807)


result = newmodel.evaluate_generator(test_generator, val_samples=807)
print newmodel.metrics_names
print result