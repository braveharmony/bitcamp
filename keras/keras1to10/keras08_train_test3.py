import tensorflow as tf
from tensorflow.keras.layers import Dense,LeakyReLU
from tensorflow.keras.models import Sequential
import matplotlib as mpl
from matplotlib import pyplot as plt


# 1. Data
import numpy as np
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([10,9,8,7,6,5,4,3,2,1,])

import random
indexset=[i for i in range(len(x))]
random.shuffle(indexset)
x_train=np.array([x[indexset[i]] for i in range(7)])
y_train=np.array([y[indexset[i]] for i in range(7)])
x_test=np.array([x[indexset[i]] for i in range(7,10)])
y_test=np.array([y[indexset[i]] for i in range(7,10)])



# from sklearn.model_selection import train_test_split #셔플에 사이킷런 사용
# x_train, x_test, y_train, y_test = train_test_split(
#     x,y,test_size=0.3,
#     random_state=234,
#     shuffle=True,
#     )

# from matplotlib import pyplot as plt
# plt.scatter(x_train,y_train,s=50,c='red',label="training data")
# plt.scatter(x_test,y_test,s=50,c='green',label='test data')
# plt.legend()
# plt.show()


# # 다른 방법으로는 아래와 같은 방법이 있다
# indexset=[i for i in range(len(x))]
# random.shuffle(indexset)
# x_train=np.array([x[indexset[i]] for i in range(7)])
# y_train=np.array([y[indexset[i]] for i in range(7)])
# x_test=np.array([x[indexset[i]] for i in range(7,10)])
# y_test=np.array([y[indexset[i]] for i in range(7,10)])


# 2. Model Build
model=Sequential()
model.add(Dense(8,input_dim=1,activation="linear"))
model.add(Dense(4,activation=LeakyReLU()))
model.add(Dense(2,activation=LeakyReLU()))
model.add(Dense(1))

# 3. Compile,Training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=10,epochs=5000)

# 4. Evaluation,Predict
from sklearn.metrics import r2_score
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'결정 계수 : {r2_score(y_test,y_predict)}')

from matplotlib import pyplot as plt
xx=np.array(np.linspace(-5,15,100))
yy=model.predict(xx)
plt.figure(2)
plt.scatter(x_train,y_train,s=50,c='red')
plt.scatter(x_test,y_test,s=50,c='green')
plt.plot(xx,yy)
plt.show()
