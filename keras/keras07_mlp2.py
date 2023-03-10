import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x = np. array(
    [[1,2,3,4,5,6,7,8,9,10],
     [1,1,1,1,2,1.3,1.4,1.5,1.6,1.4]]
).T
y = np.array([11,12,13,14,15,16,17,18,19,20])

# print(np.reshape(x,(10,2))) <-틀린 예
# print(x.shape) # (10,2) -> 2개의 특성을 가진 10개에 데이터
# print(y.shape)  # (10,) ->10개의 데이터

# 2.모델구성
model=Sequential()
model.add(Dense(5, input_dim=2))
model.add(Dense(1))

# 3.컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=2,epochs=1000)

# 4.모델 평가
loss=model.evaluate(x,y)
print(f'loss : {loss}')
result=model.predict([[10,1.4]])
print(f'[[10,1.4]]의 예측치 : {result}')

# tx=x.T
# from matplotlib import pyplot as plt
# plt.plot(tx[0],tx[1])
# plt.show()