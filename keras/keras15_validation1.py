import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import random
import matplotlib as mpl
import matplotlib.pyplot as plt

# 1. 데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))

x_val = np.array([14,15,16])
y_val = np.array([14,15,16])

x_test = np.array([11,12,13])
y_test = np.array([11,12,13])


# 2. 모델
model=Sequential()
model.add(Dense(5,activation='linear',input_dim=1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))


# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=3000,batch_size=len(x_train),validation_data=(x_val,y_val))


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

result = model.predict([17])
print(f'17의 예측값 : {result}')
