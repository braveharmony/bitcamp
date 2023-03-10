import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
import matplotlib as mpl
from matplotlib import pyplot as plt

x=np.array(np.arange(2,8*np.pi,0.5))
y=np.sin(x)

model=Sequential()
model.add(Dense(128, input_dim=1, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(128, activation=LeakyReLU()))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=100,epochs=5000)

xx=np.array(np.arange(0,10*np.pi,0.1))
yy=np.sin(xx)
loss=model.evaluate(xx,yy)
print(f'loss : {loss}')
results=model.predict(xx)

plt.plot(xx,results)
marker,stemline,baseline=plt.stem(x,y)
marker.set_color('orange')
stemline.set_visible(False)
baseline.set_visible(False)
plt.xlabel('x')
plt.ylabel('y')
plt.show()