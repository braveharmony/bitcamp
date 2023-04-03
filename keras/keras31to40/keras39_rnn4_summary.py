import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN
import numpy as np
x = np.array([[i+j for j in range(5)] for i in range(1,6)])
x=np.reshape(x,list(x.shape)+[1])

print(x.shape)
model=Sequential()
model.add(SimpleRNN(10,input_shape=list(x.shape[1:]),activation='linear'))
model.add(Dense(7))
model.add(Dense(1))
model.summary()