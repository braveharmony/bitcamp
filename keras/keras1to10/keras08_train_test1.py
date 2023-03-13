import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU
import matplotlib as mpl
from matplotlib import pyplot as plt

# 1. Data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])
x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]
print(x_train)
print(y_train)
print(x_test)
print(y_test)

# 2. Model Build
Model=Sequential()
Model.add(Dense(16,input_dim=1,activation='linear'))
Model.add(Dense(8,activation='linear'))
Model.add(Dense(4,activation='linear'))
Model.add(Dense(2,activation='linear'))
Model.add(Dense(1))

# 3. Compile, Training
Model.compile(loss='mae',optimizer='adam')
Model.fit(x_train,y_train,epochs=1000)

# 4. Model evaluation
loss=Model.evaluate(x_test,y_test)
print(f'loss : {loss}')
result=Model.predict([11])
print(f'[11]의 예측 값 : {result}')

xx=np.array(np.linspace(-5,15,1000))
yy=Model.predict(xx)
plt.scatter(x_train, y_train, s=50, c='red')
plt.scatter(x_test,y_test,s=50,c='green')
plt.plot(xx,yy)
plt.grid()
plt.show()