import tensorflow as tf
import numpy as np
import pandas as pd
import random,time,datetime

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
dataset=np.array(range(1,13))
def split_to_time(dataset,timestep):
    return np.array(list((dataset[i:i+timestep] for i in range(len(dataset)-timestep+1))))
x=np.concatenate((split_to_time(dataset,3),np.array([[20,30,40],[30,40,50],[40,50,60]])),axis=0)
y=np.array(list(range(4,14))+[50,60,70])

x_test=np.array([[50,60,70]])

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])

x=reshape(x)
x_test=reshape(x_test)

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import SimpleRNN,Dense,Bidirectional
model=Sequential((Bidirectional(SimpleRNN(10,activation='linear'),input_shape=x.shape[1:]),Dense(1)))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=3000,batch_size=len(x),verbose=True)

# 4. modelpredict
print(f'{x_test}의 결과 : {model.predict(x_test)}')