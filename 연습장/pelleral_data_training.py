import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LeakyReLU,Dense
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os,random

# 1. 데이터 생성,시드 초기화
x=np.array([np.arange(0,10,0.01)]).T
print(x.shape)
y = np.concatenate((x[int(0.5*len(x))-1::-1],x[int(0.5*len(x))-1::-1] ))
# x[:int(0.5*len(x))]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1)
seed=0
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 2. 모델 생성
model=Sequential()
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1))

# 3. 컴파일, 학습
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=1000000)

# 4. 모델 평가
loss=model.evaluate(x_test,y_test)
print(f"loss : {loss}")
result=model.predict([8])
print(f'[8]의 예측치 : {result}')

xx=np.array([np.arange(-5,15,0.2)]).T
yy=model.predict(xx)
plt.scatter(x_train,y_train,s=10,c='red')
plt.scatter(x_test,y_test,s=10,c='green')
plt.plot(xx,yy)
plt.show()