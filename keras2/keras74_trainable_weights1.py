import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU
#1. 데이터
x=np.array(range(5))+1
y=x.copy()


#2. 
model=Sequential()
model.add(Dense(16,input_dim=1,activation=LeakyReLU(0.5)))
# model.add(Dropout(1/8))
model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dropout(1/8))
model.add(Dense(16,activation=LeakyReLU(0.5)))
# model.add(Dropout(1/8))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(1))

model.layers[1].trainable=False
print(model.layers[1].trainable_weights)
model.summary()

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=5)