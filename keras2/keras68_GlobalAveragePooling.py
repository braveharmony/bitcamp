import tensorflow as tf
import random,os,numpy as np

# 0. seed initialization
seed=42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)
# 1. data prepare
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()

x_train=x_train.reshape(*x_train.shape,1)
x_test=x_test.reshape(*x_test.shape,1)

# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,GlobalAveragePooling2D,Conv2D,MaxPool2D,Flatten,AveragePooling2D,Dropout
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(64,kernel_size=(3,3),padding='same',activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='relu'))
model.add(AveragePooling2D())
model.add(Conv2D(64,kernel_size=(2,2),padding='valid',activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(10,activation='softmax'))
model.summary()

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,batch_size=100,epochs=10,validation_data=(x_test,y_test))