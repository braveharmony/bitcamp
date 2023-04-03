import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,SimpleRNN,LSTM,GRU,GRU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 0. seed
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.data prepare
datasets = np.array([i for i in range(1,13)])
x = np.array([[datasets[i],datasets[i+1],datasets[i+2]]for i in range(10)]+[[20,30,40],[30,40,50],[40,50,60]])
y = np.array([datasets[i]+3 for i in range(10)]+[50,60,70])
print(x.shape,y.shape)
x=np.reshape(x,list(x.shape)+[1])


# 2. model build
model=Sequential()
model.add(SimpleRNN(64,activation='linear',input_shape=(x.shape[1:]),return_sequences=True))
model.add(SimpleRNN(64,activation='linear',))
model.add(Dense(64,activation='linear',))
model.add(Dense(1))
model.summary()

# 3. compile,predict
model.compile(loss='mse',optimizer='adam')
hist = model.fit(x,y,epochs=600)

print(f'[[[50],[60],[70]]]의 예상값 : {model.predict(np.reshape(np.array([[50,60,70]]),list(np.array([[50,60,70]]).shape)+[1]))}')

plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()