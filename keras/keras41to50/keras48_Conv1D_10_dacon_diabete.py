# 0. seed initialization
import random
import numpy as np
import tensorflow as tf
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
df=pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0)
dft=pd.read_csv('./_data/dacon_diabete/test.csv',index_col=0)
dfs=pd.read_csv('./_data/dacon_diabete/sample_submission.csv')
print(df.info)
print(dft.info)
print(df.columns)
print(dft.columns)

x=df.drop(df.columns[-1],axis=1)
y=df[df.columns[-1]]

# print(np.unique(y))
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)
dft=reshape(dft)

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
model.add(Dense(1,activation='sigmoid'))

# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=1000
          ,batch_size=len(x_train),verbose=True,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
y_predict=np.round(model.predict(x_test))
from sklearn.metrics import accuracy_score
print(f'acc : {accuracy_score(y_test,y_predict)}')

# 5. save
y_predict=np.round(model.predict(dft))
dfs[df.columns[-1]]=y_predict
import datetime
now=datetime.datetime.now().strftime('%H시%M분')
dfs.to_csv(f'./_save/dacon_diabete/03_23/{now}_forsubConv.csv',index=False)