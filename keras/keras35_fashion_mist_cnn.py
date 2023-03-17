import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Conv2D,MaxPool2D,Flatten,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import fashion_mnist

# 0. seed initialization

seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],x_train.shape[2],1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],x_test.shape[2],1))

# print(x_train.shape,y_train.shape)
# print(np.min(x_train),np.max(x_train))
print(np.unique(y_test,return_counts=True))
y_train=np.array(pd.get_dummies(y_train,prefix='number'))/255.
y_test=np.array(pd.get_dummies(y_test,prefix='number'))/255.


# 2. model building
model=Sequential()
model.add(Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,filters=64
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation=LeakyReLU(0.5)))
model.add(Conv2D(filters=64
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation=LeakyReLU(0.5)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation=LeakyReLU(0.5)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters=256
                 ,kernel_size=(4,4)
                 ,padding='valid'
                 ,activation=LeakyReLU(0.5)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(filters=512
                 ,kernel_size=(2,2)
                 ,padding='same'
                 ,activation=LeakyReLU(0.5)))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=10000
          ,batch_size=len(x_train)//30,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
eva=model.evaluate(x_test,y_test)
print(f'loss : {eva[0]}\nacc : {eva[1]}')