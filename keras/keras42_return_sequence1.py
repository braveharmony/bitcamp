import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LSTM,SimpleRNN
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
datasets = np.array([i for i in range(1,11)])
k=5
x = np.array([[datasets[i+j] for j in range(k)] for i in range(10-k)])
y = np.array([datasets[i+k] for i in range(10-k)])
print(x.shape,y.shape)
x=np.reshape(x,list(x.shape)+[1])

# 3. model build
model=Sequential()
model.add(SimpleRNN(64,activation='linear',input_shape=(x.shape[1:]),return_sequences=True))
model.add(SimpleRNN(64,activation='linear',))
model.add(Dense(16,activation='linear',))
model.add(Dense(1))
model.summary()