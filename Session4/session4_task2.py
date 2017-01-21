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
img_path = '../../Databases/MIT_split/test/coast/art1130.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
#crop the model up to a certain layer
model = Model(input=base_model.input, output=base_model.get_layer('block3_conv2').output)
#get the features from images
features = model.predict(x)

features_3d = features[0][:]

#get rid of the third dimension
option = 'pp' # 'maximum' 'average'

if (option == 'average'):
    features_to_show=np.zeros((features_3d.shape[0],features_3d.shape[1]))
    for p in range(0,features_3d.shape[2]):
        features_to_show[:,:] = features_to_show[:,:] +  features_3d[:,:,p] 
    #average
    features_to_show = features_to_show/features_3d.shape[2];    
elif(option == 'maximum'):
    features_to_show=np.zeros((features_3d.shape[0],features_3d.shape[1]))
    for w in range(0,features_3d.shape[0]):
        for h in range(0,features_3d.shape[1]):
            features_to_show[w][h] = np.amax(features_3d[w,h,:])
else:
    print 'option not recognized'     

plt.figure()
plt.imshow(features_to_show)
plt.title('block3_conv2 Features')
plt.savefig('block3_conv2_features.png', bbox_inches='tight')

#weights = base_model.get_layer('block1_conv1').get_weights()
