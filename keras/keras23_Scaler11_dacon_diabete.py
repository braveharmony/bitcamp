import tensorflow as tf 
import numpy as np
import matplotlib as mpl
import pandas as pd
import random,time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
# print(df.info(),dft.info())
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed,stratify=y)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train);x_test=scaler.transform(x_test);dft=scaler.transform(dft)

# 2. model build
model=Sequential()
model.add(Dense(32,input_shape=(x.shape[1],),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. compile, training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
start_time=time.time()
hist=model.fit(x_train,y_train,epochs=1000
               ,batch_size=len(x),validation_split=0.2,verbose=True
               ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True))
runtime=time.time()-start_time

#4. predict,evaluate
print(f'accurancy : {accuracy_score(y_test,np.round(model.predict(x_test)))}')
dfs[df.columns[-1]]=np.round(model.predict(dft))
dfs.to_csv('./_save/dacon_diabete/03_13/forsub.csv',index=False)