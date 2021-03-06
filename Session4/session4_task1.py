from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.utils.visualize_util import plot
import numpy as np
import matplotlib.pyplot as plt
#load VGG model
base_model = VGG16(weights='imagenet')
#visalize topology in an image
plot(base_model, to_file='modelVGG16.png', show_shapes=True, show_layer_names=True)
#read and process image
img_path = '/data/MIT/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block5_conv2').output)
#get the features from images
features = model.predict(x)

weights = base_model.get_layer('block1_conv1').get_weights()

weights = weights[0]
weights_fig = np.zeros((24,24,3))

for b in range(0, weights.shape[0]):
  c = 0
  for h in range(0, weights_fig.shape[0],weights.shape[1]):
    for w in range(0, weights_fig.shape[1],weights.shape[2]):
      weights_fig[h:h+weights.shape[1],w:w+weights.shape[2],b] = weights[b,:,:,c]
      c = c + 1

plt.figure()
plt.imshow(weights_fig, interpolation="nearest")
plt.title('block1_conv1 Weights')
plt.savefig('block1_conv1_weights.png', bbox_inches='tight')