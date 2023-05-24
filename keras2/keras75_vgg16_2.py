from tensorflow.keras.applications import VGG16,ResNet50V2
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 1. dataprepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()

# 2. modelbuild
model=Sequential()
pretrained=VGG16()
pretrained.summary()
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

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test))

model=Sequential()
pretrained=VGG16()
pretrained.summary()
for layer in VGG16(include_top=False,input_shape=(32,32,3)).layers:
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

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=100,batch_size=100,validation_data=(x_test,y_test))