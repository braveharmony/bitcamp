from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
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
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test),callbacks=es)
acc_no_trained=accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1))
for i in range(len(model.layers)):
    model.layers[i].trainable=True
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test),callbacks=es)
acc_trained_after_fitting_Dense=accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1))
    
model=Sequential()
for layer in VGG16(include_top=False,input_shape=(32,32,3)).layers:
    model.add(layer)
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(512,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test),callbacks=es)
acc_train=accuracy_score(y_test,np.argmax(model.predict(x_test),axis=1))

print(f"acc with no trained : {acc_no_trained}\nacc with trained : {acc_train}\nacc with combination trained : {acc_trained_after_fitting_Dense}")

# acc with no trained : 0.6281
# acc with trained : 0.8142
# acc with combination trained : 0.82