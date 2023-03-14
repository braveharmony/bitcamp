## mlp란 multi layer perceptron을 뜻한다.


import tensorflow as tf


import numpy as np
# 1. 데이터
x = np. array(
    [[1,1],
     [2,1],
     [3,1],
     [4,1],
     [5,2],
     [6,1.3],
     [7,1.4],
     [8,1.5],
     [9,1.6],
     [10,1.4]]
)
y = np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape)  # (10,2) -> 2개의 특성을 가진 10개에 데이터
print(y.shape)  # (10,) ->10개의 데이터

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
# 2.모델구성
model=Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(5))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=2,epochs=10000)

# 4.모델 평가
loss=model.evaluate(x,y)
print(f'loss : {loss}')
result=model.predict([[10,1.4]])
print(f'[[10,1.4]]의 예측치 : {result}')

# tx=x.T
# from matplotlib import pyplot as plt
# plt.plot(tx[0],tx[1])
# plt.show()