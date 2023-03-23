import numpy as np
import tensorflow as tf
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets = load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# def split_to_time(dataset,timesteps):
#     gen=(dataset[i:i+timesteps]for i in range(len(dataset)-timesteps+1))
#     return np.array(list(gen))
# timesteps=5
# x_train=split_to_time(x_train,timesteps)
# x_test=split_to_time(x_test,timesteps)
# y_test=y_test[timesteps-1:]
# print(x_train.shape)
def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)



# 2. model build
model=Sequential(SimpleRNN(16,input_shape=x_train.shape[1:]))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(1))

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10000,batch_size=len(x_train)
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=0,verbose=True,restore_best_weights=True))

# 4. predict,evaluate
print(f'결정 계수 : {r2_score(y_test,model.predict(x_test))}')