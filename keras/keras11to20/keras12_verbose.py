# hyperparameter verbose는 진행 상황을 프린트 해주는가에 대해서 설정하는 함수이다. 0이하(False),1(auto,True),2,나머지3 이상 or double 으로 구분된다 
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing
import random

# 0.seed
seed=0
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)


# 1.데이터
datasets=fetch_california_housing()
x=datasets.data
y=np.array([datasets.target]).T
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)
print(x.shape,y.shape)

# 2.모델 생성
model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))

# 3.컴파일,훈련
verb=-False
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=10,verbose=verb)

# 4.평가,예측
loss=model.evaluate(x_test,y_test,batch_size=1000,verbose=verb)
y_predict=model.predict(x_test,verbose=verb)
r2=r2_score(y_predict,y_test)
print(f'loss : {loss}\n r2 : {r2}')