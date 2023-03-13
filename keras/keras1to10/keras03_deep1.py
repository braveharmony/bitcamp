# 1. 데이터
import numpy as np
x = np.array([1, 2, 3])
y = np.array([1, 2, 3])

# 2.모델구성
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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
# 마지막 한층에 결과값을 통합
# 위에는 전부 hidden layer를 구성하는 장치, 하지만 아직 activation function이 적용되어 있지 않은
# 상태이므로 모든 값은 linear하게 적용되며 differential value를 알 수 있어서 실질적으로hidden은 아님


# 3.컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100)
weights=model.get_weights()
# for i,weight in enumerate(weights):
#     print(f"layer {i} weights: {weight}")
    
# loss: 6.2993e-06
# layer 0's weight: [[ 0.677737   0.8900808 -0.7231195]]
# layer 1's weight: [-0.00950237 -0.00130172  0.00104286]
# layer 2's weight: [[-0.21796367 -0.66651046  0.01131786 -0.41333774]
#  [-0.9070432   0.5886682   0.8589063   0.8136844 ]
#  [-0.7010152   0.63761806 -0.87184197 -0.14250411]]
# layer 3's weight: [ 0.00174614  0.00085335 -0.00155181 -0.00083028]
# layer 4's weight: [[-0.12201829 -0.4617054   0.4974748   0.35241485 -0.10226347]
#  [ 0.53164047 -0.498188   -0.4874489   0.05575164 -0.21495652]
#  [ 0.34637123  0.15505956 -0.14725731 -0.390703    0.26179025]
#  [-0.05789151  0.02744931 -0.05992797 -0.7479888   0.2651525 ]]
# layer 5's weight: [ 0.00053594  0.00036128  0.00153616  0.00068412 -0.00069734]
# layer 6's weight: [[ 0.4393635   0.62890977 -0.7677926 ]
#  [-0.3730625   0.32418224 -0.58540326]
#  [-0.08114188 -0.69839    -0.6954073 ]
#  [ 0.85555553  0.43961012 -0.60819614]
#  [-0.385912   -0.8092656   0.54940796]]
# layer 7's weight: [ 0.00061237  0.00063142 -0.00059763]
# layer 8's weight: [[-0.4153728 ]
#  [-0.36006948]
#  [ 0.48414326]]
# layer 9's weight: [-0.0004988]

# loss: 2.2778e-05