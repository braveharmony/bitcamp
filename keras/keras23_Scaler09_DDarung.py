import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
# print(df.describe())
# print(df.isnull().sum())
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'submission.csv')
# print(df.columns,dft.columns)
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]
# print(np.unique(y))
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 2. model build
model=Sequential()
model.add(Dense(32,input_shape=(x.shape[1],),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
start_time=time.time()
model.fit(x_train,y_train,batch_size=len(x)//10,epochs=1000,verbose=True,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=True,restore_best_weights=True))
runtime=time.time()-start_time

# 4. predict, evaluate
def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))
print(f'RMSE : {RMSE(y_test,model.predict(x_test))}\nruntime : {runtime}')
dfs[df.columns[-1]]=model.predict(dft)
dfs.to_csv('./_save/DDarung/03_13/forsub.csv',index=False)