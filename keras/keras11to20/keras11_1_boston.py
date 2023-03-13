import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

# 0. 시드
seed=4
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)
# 1. 데이터
datasets=load_boston()
x=datasets.data
y=np.array([datasets.target]).T

# for i in range(len(x.T)):
#     plt.scatter(x.T[i],y,s=1)
# plt.show()
# print(datasets)
# print(datasets.feature_names)
# ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO'
#  'B' 'LSTAT']
# print(datasets.DESCR)
# print(x.shape,y.shape) (506,13),(506)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=1234)

############## [실습] ##############
# 1. train사이즈 0.7
# 2. R2 0.8 이상
####################################
# 2. 모델 구성
model=Sequential()
model.add(Dense(16,input_dim=13,activation='sigmoid'))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(1))

# 3. 컴파일,학습
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=3000)

# 4. 평가,예측
loss=model.evaluate(x_test,y_test)
print(f"loss : {loss}")
y_predict=model.predict(x_test)

r2=r2_score(y_test,y_predict)
print(f'r2_score : {r2}')

# 번외. 플로팅
atr=datasets.feature_names
tar=['boston']

plt.figure(1)
for i in range(len(atr)):
    # plt.figure(atr[i])
    plt.subplot(int(len(atr)//2.5),5,2*i+1)
    plt.scatter(x.T[i],y,s=1)
    min_x=min(x.T[i]);max_x=max(x.T[i]);min_y=min(y);max_y=max(y)
    plt.xlim(min_x-0.05*abs(max_x-min_x),max_x+0.05*abs(max_x-min_x))
    plt.ylim(min_y-0.05*abs(max_y-min_y),max_y+0.05*abs(max_y-min_y))
    plt.ylabel(tar[0],fontsize=5);plt.xlabel(atr[i],fontsize=5);plt.title(atr[i],fontsize=5)
plt.show()
