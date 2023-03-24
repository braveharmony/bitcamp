import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM,Dense,Dropout,LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
df=pd.read_csv('./_data/kaggle_jena/data.csv',index_col=0)
print(df)
print(df.columns)
print(df.info())
print(df.describe())

print(type(df[df.columns[3]].values))
import matplotlib.pyplot as plt
# plt.plot(df[df.columns[3]].values)
# plt.show()

x=df.drop(df.columns[3],axis=1)
y=df[df.columns[3]]
ts=20

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,shuffle=False)
x_test,x_predict,y_test,y_predict=train_test_split(x_test,y_test,train_size=2/3,shuffle=False)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(x_predict.shape,y_predict.shape)

def time_splitx(x,ts,scaler):
    x=scaler.transform(x)
    gen=(x[i:i+ts] for i in range(len(x)-ts+1))
    return np.array(list(gen))[:,:-1]

def time_splity(y,ts):
    gen=(y[i:i+ts] for i in range(len(y)-ts+1))
    return np.array(list(gen))[:,-1]

scaler=MinMaxScaler()
scaler.fit(x_train)

x_train=time_splitx(x_train,ts,scaler)
x_test=time_splitx(x_test,ts,scaler)
x_predict=time_splitx(x_predict,ts,scaler)

y_train=time_splity(y_train,ts)
y_test=time_splity(y_test,ts)
y_predict=time_splity(y_predict,ts)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(x_predict.shape,y_predict.shape)

# 2. model build
model=Sequential()
model.add(LSTM(10,input_shape=(x_train.shape[1:])))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(1/16))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(1/16))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(1/16))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dense(1))
model.summary()

# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train
          ,epochs=100,batch_size=len(x_train)//300
          ,validation_split=0.1,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=5
                                   ,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
from sklearn.metrics import mean_squared_error
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print(f'RMSE : {RMSE(y_test,model.predict(x_test))}')