import tensorflow as tf
import numpy as np
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
df = pd.read_csv('./_data/DDarung/train.csv',index_col=0)
dft = pd.read_csv('./_data/DDarung/test.csv',index_col=0)
dfs = pd.read_csv('./_data/DDarung/submission.csv')

df=df.dropna()
x=df.drop(df.columns[-1],axis=1)
y=df[df.columns[-1]]
print(x.columns)
print(dft.columns)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# def split_to_time(dataset,timesteps):
#     gen=(dataset[i:i+timesteps] for i in range(len(dataset)-timesteps+1))
#     return np.array(list(gen))
# timesteps=1
# x_train=split_to_time(x_train,timesteps)
# x_test=split_to_time(x_test,timesteps)
# dft=split_to_time(dft,timesteps)
# y_test=y_test[timesteps-1:]
def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)
dft=reshape(dft)

# 2. model build
model=Sequential((Conv1D(10,kernel_size=(3),padding='same',input_shape=x_train.shape[1:],activation='relu')
                  ,Flatten(),Dense(16,activation='relu'),Dense(16,activation='relu'),Dense(16,activation='relu')
                  ,Dense(1)))
model.summary()

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1000,
          batch_size=len(x_train),verbose=True
          ,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=5,verbose=True,restore_best_weights=True))

# 4. predict,evaluate
print(f'결정 계수 : {r2_score(y_test,model.predict(x_test))}')

# 5. save
y_predict=model.predict(dft)
dfs[df.columns[-1]]=y_predict
now=datetime.datetime.now().strftime('%H시%M분')
print(now)
dfs.to_csv(f'./_save/DDarung/03_23/{now}_forsubwithConv.csv')