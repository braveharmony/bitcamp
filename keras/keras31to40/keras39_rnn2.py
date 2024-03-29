import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Input,Dense,MaxPool1D,SimpleRNN
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import matplotlib.pyplot as plt

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.data prepare
datasets = np.array([i for i in range(1,11)])
k=4
x = np.array([[datasets[i+j] for j in range(k)] for i in range(10-k)])
y = np.array([datasets[i+k] for i in range(10-k)])
print(x.shape,y.shape)
x=np.reshape(x,list(x.shape)+[1])

# 2. model build
model=Sequential()
model.add(SimpleRNN(32,input_shape=list(x.shape[1:]),activation='linear'))
model.add(Dense(1,activation='linear'))
model.add(Dense(1))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=10000)

# 4. predict,evaluate
loss= model.evaluate(x,y)
x_predict = np.array([7,8,9,10]).reshape(1,4,1)
# print(x_predict.shape)

result=model.predict(x_predict)

print(f'loss : {loss}\n[7,8,9,10]의 결과 : {result}')