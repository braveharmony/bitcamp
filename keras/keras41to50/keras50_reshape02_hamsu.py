import tensorflow as tf
import random
import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Conv2D,Reshape,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.datasets import mnist
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=mnist.load_data()

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])/255.
x_train=reshape(x_train)
x_test=reshape(x_test)

def onehot(y):
    return np.array(pd.get_dummies(y,prefix='number'))
y_train=onehot(y_train)
y_test=onehot(y_test)

# 2. model build
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Conv1D,Reshape,LSTM,Input
inputlayer=Input(shape=x_train.shape[1:])
layer=Conv2D(filters=64,kernel_size=(3,3),padding='same')(inputlayer)
layer=MaxPooling2D()(layer)
layer=Conv2D(32,(3,3))(layer)
layer=Conv2D(10,3)(layer)
layer=MaxPooling2D()(layer)
layer=Flatten()(layer)    #(N,250)
layer=Reshape(target_shape=(25,10))(layer)
layer=Conv1D(10, 3)(layer)
layer=LSTM(25)(layer)
layer=Reshape(target_shape=(5,5,1))(layer)
layer=Conv2D(32,(3,3),padding='same')(layer)
layer=Flatten()(layer)
layer=Dense(100)(layer)
layer=Dense(y_train.shape[1],activation='softmax')(layer)
model=Model(inputs=inputlayer,outputs=layer)
model.summary()

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train
          ,batch_size=len(x_train)//100,epochs=200
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=5
                                   ,verbose=True,restore_best_weights=True))

# 4. predict, evaluate
from sklearn.metrics import accuracy_score
def decode(y):
    return np.argmax(y,axis=1)
print(f'acc : {accuracy_score(y_test,model.predict(x_test))}')