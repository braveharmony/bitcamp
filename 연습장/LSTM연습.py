import pandas as pd
import numpy as np
import tensorflow as tf
import random

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 1. data prepare
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
df=pd.read_csv('./_data/kaggle_jena/data.csv',index_col=0)
print(df.info())
print(df)
x=df.drop(df.columns[3],axis=1)
y=df[df.columns[3]]

# import matplotlib.pyplot as plt
# plt.plot(range(len(y[:48*6*14])),y[:48*6*14],label=df.columns[3])
# plt.legend()
# plt.show()

x_train,_,_,_=train_test_split(x,y,train_size=0.8,shuffle=False)

scaler=RobustScaler()
scaler.fit(x_train)
x=scaler.transform(x)

ts=10
# print(x.shape)

def split_to_timesteps(data,ts):
    gen=(data[i:i+ts] for i in range(len(data)-ts))
    return np.array(list(gen))
x=split_to_timesteps(x,ts)
y=y[ts:]
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,shuffle=False)
x_test,x_pred,y_test,y_pred=train_test_split(x_test,y_test,train_size=2/3,shuffle=False)

# 2. model build 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,LSTM,Dropout,Flatten
model=Sequential()
model.add(LSTM(16,input_shape=(10, 13)))
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(32,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(32,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(64,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(1,activation='linear'))
model.summary()

# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping
import time
model.compile(loss='mse',optimizer='adam')
start_time=time.time()
model.fit(x_train,y_train
          ,epochs=100,batch_size=len(x_train)//300
          ,validation_data=(x_test,y_test),verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=3,verbose=True,restore_best_weights=False))

# 4. predict,evaluate
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def RMSE(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
y_predict=model.predict(x_pred,batch_size=len(x_pred)//50,verbose=True)
print(f'결정계수 : {r2_score(y_pred,y_predict)}\nRMSE : {RMSE(y_pred,y_predict)}\nruntime : {time.time()-start_time}')