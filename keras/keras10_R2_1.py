import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


#0. 시드 초기화
seed=0
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

#1. 데이터
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20]) 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=1)

#2. 모델 구성
model=Sequential()
model.add(Dense(1,input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=20,epochs=1)

#4. 평가, 예측,r2_score
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x)

r2=r2_score(y_test,y_predict)
print(f'r2 score : {r2}')

#5. 시각화
plt.figure('결과치')
plt.scatter(x,y,s=10,c='red')
plt.plot(x,y_predict)
plt.show()