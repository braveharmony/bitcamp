import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,LeakyReLU,SimpleRNN
import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import random,time,datetime

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
def reshape(x):
    return np.reshape(x,list(x.shape[:2])+[x.shape[2]*x.shape[3]])
x_train=reshape(x_train)
x_test=reshape(x_test)
y_train=np.array(pd.get_dummies(y_train[:,0],prefix='number'))
y_test=np.array(pd.get_dummies(y_test[:,0],prefix='number'))

# 2. model build
model=Sequential(SimpleRNN(16,activation='linear',input_shape=x_train.shape[1:]))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=len(x_train)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=True,patience=5))

# 4. predict,evaluate
def RMSE(y_test,y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred=np.argmax(model.predict(x_test),axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'정확도 : {accuracy_score(y_test,y_pred)}')
