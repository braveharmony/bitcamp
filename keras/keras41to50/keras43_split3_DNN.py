import tensorflow as tf
import numpy as np
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.metrics import r2_score
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# 1. data prepare

dataset = range(1,106)

def split_x(dataset,timestep):
    gen=(dataset[i:i+timestep] for i in range(len(dataset)-timestep+1))
    return np.array(list(gen))
data=split_x(dataset,5)
print(data.shape)
x=data[:,:-1]
y=data[:,-1]
dataset_predict = range(96,106)
x_test=split_x(dataset_predict,4)
y_test=x_test[:,-1]+1

print(x.shape,y.shape)

model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=len(x),epochs=3000)

y_predict=model.predict(x_test)
print(f'결정계수 : {r2_score(y_test,y_predict)}')