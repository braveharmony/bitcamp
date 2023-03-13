import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random, time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_breast_cancer()
x=datasets.data
y=datasets.target

# print(np.unique(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,stratify=y,random_state=seed)
MMS=MinMaxScaler().fit(x_train)
x_train=MMS.transform(x_train)
x_test=MMS.transform(x_test)

# print(x_train.shape)

# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. compile, training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True)
start_time=time.time()
hist=model.fit(x_train,y_train,epochs=1,batch_size=len(x),verbose=True,validation_split=0.2,callbacks=es)
runtime=time.time()-start_time

# 4. predict
y_predict=np.round(model.predict(x_test))
print(f'accuracy : {accuracy_score(y_test,y_predict)}')
print(f'runtime : {runtime}')
