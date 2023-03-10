
import tensorflow as tf

# 1. 데이터
import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([1,2,3,5,4])

# 2. 모델 만들기
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model=Sequential()
model.add(Dense(1,input_dim=1))

# 3. 모델 실행
model.compile(loss="mae",optimizer='adam')
model.fit(x,y,epochs=10000)

# 4. 모델 평가
loss=model.evaluate(x,y)
print(f'loss : {loss}')
result=model.predict([6])
print(f'[6]의 예측값 : {result}')