import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import numpy as np
import random,time,datetime


dataset = range(1,101)
dataset_predict = range(96,106)
def split_x(dataset,timesteps):
    gen=(dataset[i : i+timesteps] for i in range(len(dataset)- timesteps + 1))#제너레이터 메소드
    return np.array(list(gen))
    return np.array([dataset[i : i+timesteps] for i in range(len(dataset)- timesteps + 1)])#리스트 컴프리헨션

def RNN_reshape(x):
    return np.reshape(x,list(x.shape)+[1])

datatarget=split_x(dataset,4+1)
x=datatarget[:,:-1]
x=RNN_reshape(x)
y=datatarget[:,-1]
x_test=split_x(dataset_predict,4)
x_test=RNN_reshape(x_test)
y_test= np.array([100+i for i in range(len(x_test))])


# model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM
model = Sequential()
model.add(LSTM(units=32,input_shape=x.shape[1:],activation='linear'))
model.add(Dense(1))

# compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=len(x)//10,epochs=1000)

# predict,evaluate
from sklearn.metrics import r2_score
y_predict=model.predict(x_test)
print(f'결정계수 : {r2_score(y_test,y_predict)}')