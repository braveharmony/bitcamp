from tensorflow.keras.applications import ResNet50,VGG19,Xception,ResNet101,InceptionV3,InceptionResNetV2,DenseNet121,MobileNetV2,NASNetMobile,EfficientNetB0
import numpy as np
import tensorflow as tf
import random,os
from tensorflow.keras.preprocessing import image
import requests
from PIL import Image
import io
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D,Dense,Dropout
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
models=[ResNet50,VGG19,Xception,ResNet101,InceptionV3,InceptionResNetV2,DenseNet121,MobileNetV2,NASNetMobile,EfficientNetB0]
for modelclass in models:
    try:
        model=Sequential()
        pretrained_model=modelclass(weights='imagenet',input_shape=(32,32,3),include_top=False)
        pretrained_model.trainable=False
        model.add(pretrained_model)
        model.add(GlobalAveragePooling2D())
        model.add(Dense(100,activation='softmax'))
    
    
        # path='D:\study_data\_data\cat_dog\PetImages\Dog\\70.jpg'
        # img=image.load_img(path,target_size=(224,224))
        # x=image.img_to_array(img)

        # x=x.reshape(-1,*x.shape)
        # x=np.expand_dims(x,axis=0)
        print('====================================')
        print(f'model:{modelclass.__name__}')
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
        model.fit(x_train,y_train,epochs=10,batch_size=100,validation_data=(x_test,y_test),)
            
    except:pass