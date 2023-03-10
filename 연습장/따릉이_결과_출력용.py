import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# 0. 시드값 정하기
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터 불러오기
path='./_data/DDarung/'
train_csv=pd.read_csv(path+'train.csv',index_col=0)
# print(train_csv)
train_csv=train_csv.dropna()
# print(train_csv.isnull().sum())


x=train_csv.drop(['count'],axis=1)
y=train_csv['count']

# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)

min_rmse=100
minflayer=0
minslayer=0
mintlayer=0
minepo=0
for first_layer in range(3,8):
    for second_layer in range(3,8):
        for third_layer in range(2,6):
            for epog in range(10,20):
                epo=200*epog
                # 2. 모델 생성
                model=Sequential()
                model.add(Dense(2**first_layer,input_dim=9,activation='sigmoid'))
                model.add(Dense(2**second_layer,activation='relu'))
                model.add(Dense(2**third_layer,activation='relu'))
                model.add(Dense(1))
                # 3. 컴파일, 훈련
                model.compile(loss='mse',optimizer='adam')
                model.fit(x_train,y_train,batch_size=len(x),epochs=epo,verbose=False)
                # 4. 평가
                y_predict=model.predict(x_test,verbose=0)
                rmse=RMSE(y_test,y_predict)
                if rmse<min_rmse:
                    min_rmse=rmse
                    minflayer=first_layer
                    minslayer=second_layer
                    mintlayer=third_layer
                    minepo=epo
                print(f'#rmse:{rmse} clay:{first_layer},{second_layer},{third_layer} epo:{epo}')
                print(f'#mrmes:{min_rmse} mlay:{minflayer},{minslayer},{mintlayer} mepo:{minepo}\n')
                
                #mrmes:48.98674093041391 mlay:4,5,2 mepo:3000