import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense,LeakyReLU
from tensorflow.keras.models import Sequential
import matplotlib as mpl
from matplotlib import pyplot as plt
import random

# 1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])
indexForSP=[i for i in range(len(x))];random.shuffle(indexForSP)
x_train = x[:7]
y_train = y[7:]
x_test = x[:7]
y_test = y[7:]
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)

# 2. Model Build
model=Sequential()
model.add(Dense(8,input_dim=1,activation='linear'))
model.add(Dense(4,activation='linear'))
model.add(Dense(2,activation='linear'))
model.add(Dense(1))

# 3. Compile,Training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=10,epochs=2000)

# 4. Evaluation,Predict
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
result=model.predict([10])
print(f'{[10]}의 예상값 : {result}')

xx=np.array(np.linspace(-5,15,100))
yy=model.predict(xx)
plt.scatter(x_train,y_train,s=50,c='red')
plt.scatter(x_test,y_test,s=50,c='green')
plt.plot(xx,yy)
plt.show()