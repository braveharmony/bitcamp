from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
# 1. dataprepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train=x_train/255.
x_test=x_test/255.

es=EarlyStopping(monitor='val_acc',mode='max',patience=5,verbose=0,restore_best_weights=True)
# 2. modelbuild
model=Sequential()
pretrained=VGG16()
for layer in VGG16(include_top=False,input_shape=(32,32,3)).layers:
    layer.trainable=False
    model.add(layer)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(100,activation='softmax'))
model.summary()
import pandas as pd
pd.set_option('max_colwidth',-1)

layers=[(layer,layer.name,layer.trainable) for layer in model.layers]
results=pd.DataFrame(layers,columns=['layer type', 'layer name', 'layer trainable'])
print(results)