import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time, datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)
y_train=np.array(pd.get_dummies(y_train.T[0],prefix='number'))
y_test=np.array(pd.get_dummies(y_test.T[0],prefix='number'))

# 2. model build
model=Sequential()
model.add(Conv2D(filters=32
                 ,kernel_size=(3,3)
                 ,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=64
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=256
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=512
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.125))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.125))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.125))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.125))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.125))
model.add(Dense(10,activation='softmax'))
model.summary()

# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=1000
          ,batch_size=len(x_train)//250,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=10,restore_best_weights=True,verbose=True)
          ,verbose=True)

# 4. predict,evaluate
date = datetime.datetime.now()
date = date.strftime('%H시%M분')
model.save(f'./_save/keras33/cifar10_{date}.h5')
print(f'accuracy : {accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1))}')