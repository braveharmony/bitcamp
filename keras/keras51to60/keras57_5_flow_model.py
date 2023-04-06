import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout,Input
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pandas as pd
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

train_generator=ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1
    ,shear_range=0.7
    ,zoom_range=0.1,
    fill_mode='nearest',
    vertical_flip=True,
    horizontal_flip=True,
)

augment_size=40000

randindx=np.random.randint(x_train.shape[0],size=augment_size)

x_train=np.reshape(x_train,list(x_train.shape)+[1])
x_test=np.reshape(x_test,list(x_test.shape)+[1])

x_argments=np.array(x_train[randindx])
y_argments=np.array(y_train[randindx])

x_argments,y_argments=train_generator.flow(x_argments,y_argments,batch_size=augment_size,shuffle=True).next()

x_train=np.concatenate((x_train/255,x_argments))
x_test=x_test/255
y_train=np.concatenate((y_train,y_argments))

len_tra=len(y_train)
y_onehot=pd.get_dummies(np.concatenate((y_train,y_test)),prefix='number')
y_train=y_onehot[:len_tra]
y_test=y_onehot[len_tra:]
del y_onehot

print(y_train.shape,y_test.shape)
# 2. model build
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(64,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(3,3),padding='same',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(128,(2,2),padding='valid',activation='relu'))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(256,(3,3),padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(1/32))
model.add(Dense(256,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(128,activation='relu'))
model.add(Dropout(1/32))
model.add(Dense(32,activation='relu'))
model.add(Dropout(1/16))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# model.fit(xy_train[0][0],xy_train[0][1],epochs=10)
hist=model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),batch_size=1000,
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