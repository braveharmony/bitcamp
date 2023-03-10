import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# 0. seed initialization
seed=0
random.seed(seed)
tf.random.set_seed(seed)
np.random.seed(seed)

# 1. data prepare
path='./_data/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'submission.csv')
# print(df)
# print(df.info())
df=df.dropna()
# print(df.isnull().sum())
# print(df.columns)
# print(dft.columns)
# print(dfs.columns)
x=df.drop([df.columns[-1]],axis=1)
# print(x)
y=df[df.columns[-1]]
# print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed)


# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(28,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='linear'))

# 3. compile,training
es=EarlyStopping(monitor='val_loss',mode='min',patience=100,restore_best_weights=True,verbose=True)
model.compile(loss='mse',optimizer="adam")
model.fit(x_train,y_train,batch_size=len(x_train)//10,verbose=True,validation_split=0.2,callbacks=es,epochs=3000)


# 4. evaluate, save
print(f'loss : {model.evaluate(x_test,y_test)}')
y_predict=model.predict(x_test)
print(f'rmse : {RMSE(y_test,y_predict)}')

y_predict=model.predict(dft)
# print(y_predict)
dfs[df.columns[-1]]=y_predict
# print(dfs)
pathsave='./_save/DDarung/'
dfs.to_csv(pathsave+'submission_03_08.csv',index=False)