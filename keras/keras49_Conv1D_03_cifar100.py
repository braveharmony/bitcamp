import random
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv1D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
import pandas as pd
from tensorflow.python.keras.datasets import cifar100


# 0. seed initializtion
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
y_train=np.array(pd.get_dummies(y_train[:,0],prefix='number'))
y_test=np.array(pd.get_dummies(y_test[:,0],prefix='number'))

def reshape(x):
    return np.reshape(x,list(x.shape[:2])+[x.shape[2]*x.shape[3]])
x_train=reshape(x_train)
x_test=reshape(x_test)

# 2. model build

model=Sequential()
model.add(Conv1D(10,kernel_size=(3),padding='same',activation='relu',input_shape=x_train.shape[1:]))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=1
          ,batch_size=len(x_train),verbose=True,validation_split=0.2
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
y_predict=np.argmax(model.predict(x_test),axis=1)
y_test=np.argmax(y_test,axis=1)
from sklearn.metrics import accuracy_score
print(f'acc : {accuracy_score(y_test,y_predict)}')