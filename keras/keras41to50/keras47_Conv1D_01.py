import numpy as np



#2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten
model=Sequential((Conv1D(10,2,input_shape=(3,1)),Flatten(),Dense(5),Dense(1)))
model.summary()