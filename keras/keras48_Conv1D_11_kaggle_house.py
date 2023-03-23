import tensorflow as tf
import pandas as pd
import numpy as np
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout,Conv2D,Flatten,SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# 1. data prepare
path = './_data/kaggle_house/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
# print(df.info())
# print(dft.info())
# print(type(df[df.columns[2]][1]),type(str()))
le=LabelEncoder()
for i in dft.columns:
    if dft[i].dtypes=='object':
        dft[i]=le.fit_transform(dft[i])
        df[i]=le.fit_transform(df[i])
# print(df.info())
# print(dft.info())
# print(dfs.info())

df=df.dropna()
print(df.info())
# 아무튼 정제
x=df.drop([dfs.columns[-1]],axis=1)
y=df[dfs.columns[-1]]
print(x.shape,y.shape)
print(np.unique(y))
print(dfs.shape,dft.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

print(x_train.shape)

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)
dft=reshape(dft)
print(dft.shape)


# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv1D,Dense,Flatten
model=Sequential()
model.add(Conv1D(10,kernel_size=(3),padding='same',activation='relu',input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))

# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000
          ,batch_size=len(x_train),verbose=True,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
y_predict=model.predict(x_test)
from sklearn.metrics import r2_score
print(f'acc : {r2_score(y_test,y_predict)}')

# 5. save
y_predict=model.predict(dft)
dfs[df.columns[-1]]=y_predict
import datetime
now=datetime.datetime.now().strftime('%H시%M분')
dfs.to_csv(f'./_save/kaggle_house/03_23/{now}_forsubConv.csv')