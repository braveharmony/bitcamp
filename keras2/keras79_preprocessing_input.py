from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
import numpy as np
import tensorflow as tf
import random,os
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import io


# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

model = ResNet50(weights='imagenet')

path='D:\study_data\_data\cat_dog\PetImages\Dog\\70.jpg'
img=image.load_img(path,target_size=(224,224))
x=image.img_to_array(img)

# x=x.reshape(-1,*x.shape)
x=np.expand_dims(x,axis=0)

print(x.shape)

print('====================================')
x=preprocess_input(x)
print(x.shape)
print(np.min(x),np.max(x))

print('====================================')
x_pred=model.predict(x)
print(decode_predictions(x_pred))