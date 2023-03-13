# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2.모델구성
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(3,input_dim=1))
# 인풋 디멘션 1개-> 3개로 발산
model.add(Dense(4))
# 윗층에서 4개의 결과값에 발산
model.add(Dense(5))
# 윗층에서 5개의 결과값에 발산
model.add(Dense(3))
# 윗층에서 3개의 결과값에 발산
model.add(Dense(1))

# 3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=2000)
weights=model.get_weights()

# 4.평가, 예측
loss = model.evaluate(x, y)
print(f'loss : {loss}')
# for i,weight in enumerate(weights):
#     print(f'layer {i} weight : {weight}')
result=model.predict([4])
print(f'[4]의 예측값 : {result}')

# 노드 1개 2만번 반복훈련
# loss : 0.0
# layer 0 weight : [[1.]]
# layer 1 weight : [4.2207596e-08]
# [4]의 예측값 : [[4.]]

# 노드 10개 2000번 반복훈련
# loss : 6.688575387991946e-11
# layer 0 weight : [[0.61465776]]
# layer 1 weight : [-0.0211454]
# layer 2 weight : [[0.84407365]]
# layer 3 weight : [-0.00747793]
# layer 4 weight : [[1.9060888]]
# layer 5 weight : [-0.00234921]
# layer 6 weight : [[0.7212801]]
# layer 7 weight : [0.00948182]
# layer 8 weight : [[-1.2318082]]
# layer 9 weight : [-0.01521852]
# layer 10 weight : [[-1.1381298]]
# layer 11 weight : [0.02059777]
# 1/1 [==============================] - 0s 82ms/step
# [4]의 예측값 : [[3.9999814]]

# 노드 20개
# loss : 2.9605946193960245e-14
# [4]의 예측값 : [[4.0000005]]

# 노드 100개
# loss : 0.6666666865348816
# [4]의 예측값 : [[2.]]

# 노드 200개
# loss : 0.6666666865348816
# [4]의 예측값 : [[2.]]