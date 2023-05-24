import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU
#1. 데이터
x=np.array(range(5))+1
y=x.copy()


#2. 
model=Sequential()
model.add(Dense(4,input_dim=1,activation=LeakyReLU(0.5)))
# model.add(Dropout(1/8))
model.add(Dense(4,activation=LeakyReLU(0.5)))
model.add(Dense(1))

model.summary()

model.trainable=False
# model.trainable=True

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,epochs=100,batch_size=5)
print(model.trainable_weights)