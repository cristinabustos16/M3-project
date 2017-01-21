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

print features1.shape

x = np.reshape(features1, (features1.shape[1] * features1.shape[2], features1.shape[3]))

print x.shape

for i in range(features1.shape[1]):
    for j in range(features1.shape[2]):
        a = features1[0, i, j, :]
        b = x[i*features1.shape[2] + j]
        z = abs(a-b)
        print sum(z)



