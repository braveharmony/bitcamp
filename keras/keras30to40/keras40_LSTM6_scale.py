import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Input,Dense,MaxPool1D,SimpleRNN,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import matplotlib.pyplot as plt

print(tf.__version__)
# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.data prepare
datasets = np.array([i for i in range(1,13)])
x = np.array([[datasets[i],datasets[i+1],datasets[i+2]]for i in range(10)]+[[20,30,40],[30,40,50],[40,50,60]])
y = np.array([datasets[i]+3 for i in range(10)]+[50,60,70])
print(x.shape,y.shape)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
x=scaler.fit_transform(x)
x=np.reshape(x,list(x.shape)+[1])



# 2. model build
model=Sequential()
model.add(LSTM(512,input_shape=list(x.shape[1:]),activation='linear'))
model.add(Dense(128,activation='linear'))
model.add(Dense(1))
model.summary()

# 3. compile,predict
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x,y,epochs=2000)

print(f'[[[50],[60],[70]]]의 예상값 : {model.predict(np.reshape(MinMaxScaler(np.array([[50,60,70]])),list(np.array([[50,60,70]]).shape)+[1]))}')

plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()