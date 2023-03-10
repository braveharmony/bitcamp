import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU
import matplotlib as mpl
from matplotlib import pyplot as plt

# 1. Data
x_train = np.array([range(10)]).T
y_train = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [1+0.1*i for i in range(10)],
                    [9-i for i in range(10)]]).T

# 2. Model Build
model=Sequential()
model.add(Dense(16,input_dim=1))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(2))
model.add(Dense(3))

# 3. Compile, fitting
model.compile(loss="mse",optimizer='adam')
model.fit(x_train,y_train,batch_size=10,epochs=1000)

# 4. Model evaluation
loss=model.evaluate(x_train,y_train) # 트레이닝 데이터를 평가에 쓰면 안된다! 따로 evaluate data set을 만들어야 한다.
print(f'loss : {loss}')
result=model.predict([[9]])
print(f'[[9]]의 예측값 : {result}')

# etc. ploting

xx=np.array([np.arange(0,11,0.1)]).T
yy=model.predict(xx).T
for i in range(len(yy)):
    plt.plot(xx,yy[i])
plt.grid()
plt.show()