import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from keras.datasets import cifar100

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
 
# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))
y_train=np.reshape(y_train,(y_train.shape[0],-1))
y_test=np.reshape(y_test,(y_test.shape[0],-1))
y_train=np.array(pd.get_dummies(y_train.T[0],prefix='number'))
y_test=np.array(pd.get_dummies(y_test.T[0],prefix='number'))


# 2. model build
model=Sequential()
model.add(Dense(512, input_shape=(x_train.shape[1],),activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(256, activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(512,activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(256,activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(128,activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(64,activation=LeakyReLU(0.125)))
model.add(Dropout(0.0625))
model.add(Dense(units=y_train.shape[1],activation='softmax'))

# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start_time=time.time()
model.fit(x_train,y_train,epochs=10000
          ,verbose=True,validation_split=0.2
          ,batch_size=len(x_train)//250
          ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=50,restore_best_weights=True,verbose=True))
runtime=time.time()-start_time
# 4. model predict
eve=model.evaluate(x_test,y_test)
print(f'acc : {eve[1]}\nruntime : {runtime}')