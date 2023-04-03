# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset
import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler
import time

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare

save_start=time.time()
path='d:/study_data/_save/cat_dog/'
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

print(x_train.shape[1:])
# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,Conv2DTranspose,MaxPool2D,Dropout,Input
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(128,(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPool2D(3,3))
model.add(Conv2D(256,(2,2),padding='valid',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(256,(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(512,(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(256,(2,2),padding='valid',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256,activation=LeakyReLU(0.75)))
model.add(Dropout(1/32))
model.add(Dense(256,activation=LeakyReLU(0.75)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.75)))
model.add(Dropout(1/32))
model.add(Dense(32,activation=LeakyReLU(0.75)))
model.add(Dropout(1/16))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# model.fit(xy_train[0][0],xy_train[0][1],epochs=10)
hist=model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),batch_size=50,
                    callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=3,restore_best_weights=True,verbose=True))

val_acc_index=hist.history['val_acc'].index(max(hist.history['val_acc']))
print('loss:',hist.history['loss'][val_acc_index])
print('val_loss:',hist.history['val_loss'][val_acc_index])
print('acc:',hist.history['acc'][val_acc_index])
print('val_acc:',hist.history['val_acc'][val_acc_index])

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()