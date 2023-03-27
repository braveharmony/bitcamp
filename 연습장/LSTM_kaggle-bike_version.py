import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,LSTM
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# 0. seed initialization
seed =0 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
df=pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0)
# print(df) # 1시간 간격
x=df.drop(df.columns[-1],axis=1)
y=df[[df.columns[-1]]]
x_train,_,_,_=train_test_split(x,y,train_size=0.8,shuffle=False)

scaler=RobustScaler()
scaler.fit(x_train)
x=scaler.transform(x)

print(x.shape)

def split_to_timesteps(data,ts):
    gen=(data[i:i+ts] for i in range(len(data)-ts))
    return np.array(list(gen))
ts=24
x=split_to_timesteps(x,ts)
y=y[ts:]

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True)
print(x_train.shape)
# 2. model build
model=Sequential()
model.add(LSTM(32,input_shape=x_train.shape[1:],activation='tanh'))
model.add(Dense(16,activation='relu'
                # ,input_shape=x_train.shape[1:]
                ))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

# 3. compile,training
import time
model.compile(loss='mse',optimizer='adam')
start=time.time()
model.fit(x_train,y_train
          ,epochs=10000,batch_size=len(x_train)//100
          ,validation_split=0.2,verbose=True,shuffle=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min'
                                   ,patience=20,verbose=True,restore_best_weights=True))

# 4. predict,evaluate
from sklearn.metrics import r2_score
print(f'runtime : {time.time()-start}\n결정계수 : {r2_score(y_test,model.predict(x_test,batch_size=len(x_test)//10,verbose=True))}')