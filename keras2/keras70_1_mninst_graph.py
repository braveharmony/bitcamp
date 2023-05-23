from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Dense,Conv2D,Flatten,MaxPooling2D,GlobalAveragePooling2D,Dropout
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.keras.callbacks import EarlyStopping,History
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
import tensorflow as tf
import random,time
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=mnist.load_data()

scaler=StandardScaler()
x_train:np.ndarray=x_train.reshape(-1,1)/255.
x_train=scaler.fit_transform(x_train)
x_test:np.ndarray=x_test.reshape(-1,1)/255.
x_test=scaler.transform(x_test)

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. model build
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(64,(2,2),padding='Valid',activation='relu'))
model.add(MaxPooling2D())
model.add(GlobalAveragePooling2D())
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/8))
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/8))
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/8))
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/8))
model.add(Dense(10,activation='softmax'))
model.summary()

# 3. compile,training
start_time=time.time()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
hist = model.fit(x_train,y_train,epochs=20,batch_size=100,verbose=True,
          validation_data=(x_test,y_test),
          callbacks=EarlyStopping(monitor='val_acc',mode='max',
                                  patience=20,verbose=True,
                                  restore_best_weights=True))

import joblib
joblib.dump(hist.history, './_save/keras70_1_history.dat')

import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid
plt.title('loss')
plt.xlabel('loss')
plt.ylabel('epochs')
plt.legend(loc ='upper right')

plt.subplot(2,1,2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid
plt.title('acc')
plt.xlabel('acc')
plt.ylabel('epochs')
plt.legend(['acc','val_acc'])
plt.show()