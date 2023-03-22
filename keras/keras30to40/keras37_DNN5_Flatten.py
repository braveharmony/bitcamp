import tensorflow as tf
import numpy as np
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten,LeakyReLU
from keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.metrics import accuracy_score
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1. data prepare
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=np.reshape(x_train,(x_train.shape[0],-1))/255.
x_test=np.reshape(x_test,(x_test.shape[0],-1))/255.
y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))
print(y_train.shape,np.unique(y_train))

#2. model build
model=Sequential()
model.add(Dense(10,input_shape=(x_train.shape[1:])))
model.add(Dense(9))
model.add(Dense(8))
model.add(Dense(7))
model.add(Flatten())
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
start_time=time.time()
hist=model.fit(x_train,y_train,epochs=10000
        ,batch_size=len(x_train)//50,verbose=True
        ,validation_split=0.2
        ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=50,restore_best_weights=True,verbose=True))
runtime=time.time()-start_time

# 4. evaluate,predict
acc=accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1))
print(f'accuaracy : {acc}\nruntime : {runtime}')