import tensorflow as tf
import numpy as np
import pandas as pd 
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.datasets import load_boston

# 0. seed initialization 
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
MMS=MinMaxScaler().fit(x_train)
x_train=MMS.transform(x_train)
x_test=MMS.transform(x_test)

# 2. model build
model=Sequential()
model.add(Dense(1,input_shape=(x.shape[1],)))

# 데이터가 3차원이면(시계열 데이터)
# (1000,100,1)->>>input_shape(100,1)
# 데이터가 4차원이면(이미지 데이터)
# (60000,32,32,3)->>>input_shape(32,32,3)

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
es=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True)
start_time=time.time()
model.fit(x_train,y_train,epochs=1,batch_size=len(x),validation_split=0.2,verbose=True,callbacks=es)
runtime=time.time()-start_time

# 4. evaluate,predict
print(f'결정 계수 : {r2_score(y_test,model.predict(x_test))}\nruntime : {runtime}')