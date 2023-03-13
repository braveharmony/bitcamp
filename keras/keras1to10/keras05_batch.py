import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# batch란 한번에 훈련할 수 있는 데이터 수가 너무 많을 경우 나눠서 batch단위로 쪼개서 연산하는것을 뜻한다.

# 1. 데이터
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

# 2. 모델구성
model=Sequential()
model.add(Dense(1,input_dim=1))

# 3. 컴파일, 훈련
model.compile(loss="mae", optimizer='adam')
model.fit(x, y, batch_size=1, epochs=10000)

# 4. 모델 평가
loss=model.evaluate(x,y)
print(f'loss : {loss}')
result=model.predict([6])
print(f'[6]의 예측값 : {result}')

# batch_size:
# Integer or None. Number of samples per batch of computation.
# If unspecified, batch_size will default to 32. Do not specify the batch_size if your data 
# is in the form of a dataset, generators, or keras.utils.Sequence instances (since they generate batches).