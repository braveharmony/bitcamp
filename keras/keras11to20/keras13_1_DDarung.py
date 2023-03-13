import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import random,math
# ###########################################################################
# 0. seed
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# ###########################################################################
# 1. 데이터
path="./_data/DDarung/"
path_save='./_save/DDarung/'
train_csv=pd.read_csv(path+'train.csv',index_col=0)
keys=train_csv.keys()

# print(train_csv)
# print(type(train_csv)) # <class 'pandas.core.frame.DataFrame'>
# print(train_csv.shape) # (1459, 10)

# test_csv=pd.read_csv(path+'test.csv',index_col=0)
# print(test_csv.shape) # (715, 9)

# ###########################################################################

# print(train_csv.columns)
# Index(['hour', 'hour_bef_temperature', 'hour_bef_precipitation',
#        'hour_bef_windspeed', 'hour_bef_humidity', 'hour_bef_visibility',
#        'hour_bef_ozone', 'hour_bef_pm10', 'hour_bef_pm2.5', 'count'],
#       dtype='object')
print(train_csv.info())
#  0   hour                    1459 non-null   int64
#  1   hour_bef_temperature    1457 non-null   float64
#  2   hour_bef_precipitation  1457 non-null   float64
#  3   hour_bef_windspeed      1450 non-null   float64
#  4   hour_bef_humidity       1457 non-null   float64
#  5   hour_bef_visibility     1457 non-null   float64
#  6   hour_bef_ozone          1383 non-null   float64
#  7   hour_bef_pm10           1369 non-null   float64
#  8   hour_bef_pm2.5          1342 non-null   float64
#  9   count                   1459 non-null   float64
print(train_csv.describe())
############################## 결측치 처리 ###################################
# 결측치 처리 1. 제거
print(train_csv.isnull().sum())
train_csv=train_csv.dropna() #nan 값 전부 밀어버림
print(train_csv.isnull().sum())
print(train_csv.info())
print(train_csv.shape)      # (1328,10)


###################### train_csv데이터에서 x와 y를 분리 #######################
x=np.array(train_csv.drop(['count'], axis=1))
y=train_csv['count']
###################### train_csv데이터에서 x와 y를 분리 #######################

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)

print(f'{x_train.shape} {x_test.shape}') # (1021, 9) (438, 9) -> (929, 9) (399, 9)
print(f'{y_train.shape} {y_test.shape}') # (1021,) (438,) -> (929,) (399,)

# 2. 모델 구성
model=Sequential()
model.add(Dense(16,input_dim=9,activation='linear'))
model.add(Dense(16,activation='linear'))
model.add(Dense(16,activation='linear'))
model.add(Dense(1))
# 3. 컴파일,훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=len(x_train),epochs=5000,verbose=True)

# 4. 평가, 예측
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')

for i in range(len(x[0])):
    # plt.figure(keys[i])
    plt.subplot(int(len(x[0])//2.5)+1,5,2*i+1)
    plt.scatter(x.T[i],y,s=1)
    min_x=min(x.T[i]);max_x=max(x.T[i]);min_y=min(y);max_y=max(y)
    plt.xlim(min_x-0.05*abs(max_x-min_x),max_x+0.05*abs(max_x-min_x));plt.xlabel(keys[i],fontsize=8)
    plt.ylim(min_y-0.05*abs(max_y-min_y),max_y+0.05*abs(max_y-min_y));plt.ylabel(keys[9],fontsize=8)
    plt.title(keys[i],fontsize=8)
plt.show()
