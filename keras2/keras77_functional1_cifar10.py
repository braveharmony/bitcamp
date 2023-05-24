from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten,Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
# 1. dataprepare
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
x_train=x_train/255.
x_test=x_test/255.

es=EarlyStopping(monitor='val_acc',mode='max',patience=5,verbose=0,restore_best_weights=True)
# 2. modelbuild
model=Sequential()
input_layer=Input(shape=(32,32,3))
layer1=input_layer
for layer in VGG16(include_top=False,input_shape=(32,32,3)).layers:
    layer.trainable=False
    layer1=layer(layer1)
layer1=Flatten()(layer1)
layer1=Dense(512,activation='relu')(layer1)
layer1=Dropout(1/16)(layer1)
layer1=Dense(512,activation='relu')(layer1)
layer1=Dropout(1/16)(layer1)
layer1=Dense(512,activation='relu')(layer1)
layer1=Dropout(1/16)(layer1)
output_layer=Dense(100,activation='softmax')(layer1)
model=Model(inputs=input_layer,outputs=output_layer)
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test),callbacks=es)
acc_no_trained=accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1))

print(f"acc with no trained : {acc_no_trained}")

