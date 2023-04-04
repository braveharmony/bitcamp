import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# 0. seed initialization 
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='d:/study_data/_save/keras58/_2/'
x_train=np.load(f'{path}x_train.npy')
x_test=np.load(f'{path}x_test.npy')
y_train=np.load(f'{path}y_train.npy')
y_test=np.load(f'{path}y_test.npy')


print(x_train.shape,y_train.shape)
gen2=ImageDataGenerator(rescale=1)
xy_train=gen2.flow(x_train,y_train,batch_size=200,shuffle=False)


# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv2D,MaxPool2D,Flatten,Dense,Dropout,LeakyReLU
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(128,(3,3),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(256,(3,3),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(256,(2,2),padding='valid',activation=LeakyReLU(0.25)))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(256,(3,3),padding='valid',activation=LeakyReLU(0.25)))
model.add(Flatten())
model.add(Dense(256,activation=LeakyReLU(0.25)))
model.add(Dropout(1/32))
model.add(Dense(256,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/32))
model.add(Dense(32,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# model.fit(xy_train[0][0],xy_train[0][1],epochs=10)
hist=model.fit(xy_train,epochs=1000,validation_data=(x_test,y_test),
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