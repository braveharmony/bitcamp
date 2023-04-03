import tensorflow as tf
import numpy as np
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.datasets import fetch_california_housing
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)
print(x_train.shape)

# 2. model build
model=Sequential((Conv1D(10,kernel_size=(3),padding='same',input_shape=x_train.shape[1:],activation='relu')
                  ,Flatten(),Dense(16,activation='relu'),Dense(16,activation='relu'),Dense(16,activation='relu')
                  ,Dense(1)))
model.summary()

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,
          batch_size=len(x_train),verbose=True
          ,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=5,verbose=True,restore_best_weights=True))

# 3. predict,evaluate
print(f'결정 계수 : {r2_score(y_test,model.predict(x_test))}')