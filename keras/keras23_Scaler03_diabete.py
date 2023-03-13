import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
MMS=MinMaxScaler().fit(x_train)
x_train=MMS.transform(x_train);x_test=MMS.transform(x_test)

# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
es=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,patience=50)
start_time=time.time()
hist=model.fit(x_train,y_train,batch_size=len(x),verbose=True,validation_split=0.2,callbacks=es,epochs=1)
runtime=time.time()-start_time

# 4. evaluate, predict
print(f'loss : {model.evaluate(x_test,y_test)}')
print(f'결정계수 : {r2_score(y_test,model.predict(x_test))}')
print(f'runtime : {runtime}')