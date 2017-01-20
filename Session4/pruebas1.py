from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras import backend as K
from keras.utils.visualize_util import plot
import numpy as np
import matplotlib.pyplot as plt
#load VGG model
base_model = VGG16(weights='imagenet')
#read and process image
img_path = '/data/MIT/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)


model1 = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)
model2 = Model(input=base_model.input, output=base_model.get_layer('block5_conv3').input)

plot(model1, to_file='model1.png', show_shapes=True, show_layer_names=True)
plot(model2, to_file='model2.png', show_shapes=True, show_layer_names=True)

#get the features from images
features1 = model1.predict(x)
features2 = model1.predict(x)

print features1.__class__.__name__
print len(features1)
print features1.shape

print sum(sum(sum(sum(features1))))
print sum(sum(sum(sum(features2))))