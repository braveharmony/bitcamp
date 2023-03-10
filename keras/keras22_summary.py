import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# 1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

# 2. 모델구성
model=Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))
model.summary()

# 3. 컴파일,훈련
# model.compile(loss='mae',optimizer='adam')
# model.fit(x,y,batch_size=len(x),validation=0.2,verbose=True)