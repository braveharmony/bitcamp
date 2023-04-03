import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_covtype
from matplotlib import pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=fetch_covtype()
x=datasets.data
y=datasets.target
y=np.array(pd.get_dummies(y,prefix='number'))
print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed,stratify=y)

# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=True,patience=50)
start_time=time.time()
model.fit(x_train,y_train,epochs=1,batch_size=len(x)//30,validation_split=0.2,verbose=True,callbacks=es)
runtime=time.time()-start_time

# 4. predict,evaluate
y_predict=np.argmax(model.predict(x_test),axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')
print(f'runtime : {runtime}')
