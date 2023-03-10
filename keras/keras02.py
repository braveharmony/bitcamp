#1. 데이터
import numpy as np
x=np.array([1,2,3])
y=np.array([1,2,3])

#2. 모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)
weights = model.get_weights()

for i, w in enumerate(weights):
    print(f"Layer {i} weights: {w}")
    
# loss: 0.0000e+00
# Layer 0 weights: [[1.]]
# Layer 1 weights: [2.8804774e-08]

