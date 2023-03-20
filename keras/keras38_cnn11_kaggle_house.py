import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout,Conv2D,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
for i in df.columns:
    if df[i].dtypes=='object':
        dft=dft.drop([i],axis=1)
        df=df.drop([i],axis=1)
# print(df.info())
# print(dft.info())
# print(dfs.info())

df=df.dropna()
print(df.info())
# 아무튼 정제
x=df.drop([dfs.columns[-1]],axis=1)
y=df[dfs.columns[-1]]
# print(x.shape,y.shape)
# print(np.unique(y))
# print(dfs.shape,dft.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
print(x_train.shape,x_test.shape)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]//6,6,1))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]//6,6,1))
print(x_train.shape,x_test.shape)



# 2. model build
model=Sequential()
model.add(Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,filters=64
                 ,kernel_size=(3,3)
                 ,padding='valid'
                 ,activation='relu'))
model.add(Conv2D(filters=64
                 ,kernel_size=(2,2)
                 ,strides=2
                 ,activation='relu'))
model.add(Conv2D(filters=64
                 ,kernel_size=(2,2)
                 ,padding='valid'
                 ,activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(1))
model.summary()


# 3. compile,fit
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=len(x),epochs=10000,validation_split=0.2,
        callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=600,restore_best_weights=True))

# 4. evaluate,predict
from sklearn.metrics import r2_score
print(f'결정계수 : {r2_score(y_test,model.predict(x_test))}')
# y_predict=np.array(model.predict(dft))
# def renonemean(arr):
#     arr = np.array(arr)
#     arr = np.where(arr == None, np.nan, arr)
#     mean = np.nanmean(arr)
#     arr = np.where(np.isnan(arr), mean, arr)
#     return arr
# dfs[dfs.columns[-1]]=renonemean(y_predict)
# # dfs.to_csv(f'./_save/kaggle_house/03_17/forsub{j}.csv',index=False)