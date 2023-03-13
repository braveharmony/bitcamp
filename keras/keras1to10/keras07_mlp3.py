import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU

# 1. 데이터
x = np. array(
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4],
     [9,8,7,6,5,4,3,2,1,0]]
).T
y = np.array([11,12,13,14,15,16,17,18,19,20])

# 실습
model=Sequential()
model.add(Dense(16,input_dim=3,activation='linear'))
model.add(Dense(1))

model.compile(loss='mae',optimizer='adam')
model.fit(x,y,batch_size=10,epochs=500)

# 예측[10,1.4,0]
loss=model.evaluate(x,y,batch_size=2)
print(f'loss : {loss}')
result=model.predict([[10,1.4,0],[1,1,9]],batch_size=1)
print(f'[[10,1.4,0],[1,1,9]]의 예측치 : {result}')