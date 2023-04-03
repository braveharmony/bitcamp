# 0. seed initializtion
import random
import numpy as np
import tensorflow as tf
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

datasets=load_iris()
x=datasets.data
y=datasets.target
y=np.array(pd.get_dummies(y,prefix='number'))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten
model=Sequential()
model.add(Conv1D(10,kernel_size=(3),padding='same',activation='relu',input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=1
          ,batch_size=len(x_train),verbose=True,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
y_predict=np.argmax(model.predict(x_test),axis=1)
y_test=np.argmax(y_test,axis=1)
from sklearn.metrics import accuracy_score
print(f'acc : {accuracy_score(y_test,y_predict)}')