####### pruebas (xian)

from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.layers import Dense
from keras.layers import Flatten

#load VGG model
base_model = VGG16(weights='imagenet')
#visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)

#crop the model up to a certain layer
model1 = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)
plot(model1, to_file='model1.png', show_shapes=True, show_layer_names=True)

#crop the model up to a certain layer
x = base_model.get_layer('block5_conv2').output
model2 = Model(input=base_model.input, output=x)
plot(model2, to_file='model2.png', show_shapes=True, show_layer_names=True)

# crop below
x = base_model.get_layer('block4_conv2').output
model3 = Model(input=base_model.input, output=x)
plot(model3, to_file='model3.png', show_shapes=True, show_layer_names=True)

# crop below and add fully connected
x = base_model.get_layer('block4_conv2').output
x = Dense(1000, activation='relu',name='predictions')(x)
model4 = Model(input=base_model.input, output=x)
plot(model4, to_file='model4.png', show_shapes=True, show_layer_names=True)

# crop below and add fully connected
#x = base_model.get_layer('block4_conv2').output
#model5 = Model(input=base_model.input, output=x)
#model5.add(Dense(1000, activation='relu',name='predictions'))
#plot(model5, to_file='model5.png', show_shapes=True, show_layer_names=True)

# crop below and add fully connected, with flatten before
x = base_model.get_layer('block4_conv2').output
x = Flatten(name='aplanado')(x)
x = Dense(1000, activation='relu',name='predictions')(x)
model6 = Model(input=base_model.input, output=x)
plot(model6, to_file='model6.png', show_shapes=True, show_layer_names=True)