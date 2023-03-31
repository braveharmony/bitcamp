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

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. x_train prepare
train_datagen=ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(
    rescale=1/255
)

xy_train=train_datagen.flow_from_directory('d:/study_data/brain/train/'
                                  ,target_size=(100,100)
                                  ,batch_size=5
                                  ,class_mode='categorical'
                                  ,color_mode='grayscale'
                                #   ,color_mode='rgb'
                                  ,shuffle=True
                                  )

xy_test=test_datagen.flow_from_directory('d:/study_data/brain/test/'
                                  ,target_size=(100,100)
                                  ,batch_size=5
                                  ,class_mode='categorical'
                                  ,color_mode='grayscale'
                                  ,shuffle=True
                                  )

x_train=(xy_train[i][0] for i in range(len(xy_train)))
y_train=(xy_train[i][1] for i in range(len(xy_train)))
x_train=np.array(list(x_train))
x_train=np.reshape(x_train,([x_train.shape[0]*x_train.shape[1]]+list(x_train.shape[2:])))
y_train=np.array(list(y_train))
y_train=np.reshape(y_train,([y_train.shape[0]*y_train.shape[1]]+list(y_train.shape[2:])))
x_test=(xy_test[i][0] for i in range(len(xy_test)))
y_test=(xy_test[i][1] for i in range(len(xy_test)))
x_test=np.array(list(x_test))
x_test=np.reshape(x_test,([x_test.shape[0]*x_test.shape[1]]+list(x_test.shape[2:])))
y_test=np.array(list(y_test))
y_test=np.reshape(y_test,([y_test.shape[0]*y_test.shape[1]]+list(y_test.shape[2:])))
print(x_train.shape)
print(y_train.shape)
print(y_train[:10])

path='d:/study_data/_save/_npy/'
np.save(file=f'{path}keras55_1_x_train.npy',arr=x_train)
np.save(file=f'{path}keras55_1_x_test.npy',arr=x_test)
np.save(file=f'{path}keras55_1_y_train.npy',arr=y_train)
np.save(file=f'{path}keras55_1_y_test.npy',arr=y_test)

'''
# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Flatten,Conv2DTranspose,MaxPool2D,Dropout
model=Sequential()
model.add(Conv2D(128,(3,3),padding='same',input_shape=x_train.shape[1:],activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(256,(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(512,(3,3),padding='valid',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(256,(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPool2D())
model.add(Conv2D(512,(2,2),padding='valid',activation=LeakyReLU(0.75)))
model.add(Conv2D(512,(2,2),padding='valid',activation=LeakyReLU(0.75)))
model.add(Flatten())
model.add(Dense(128,activation=LeakyReLU(0.75)))
model.add(Dropout(1/32))
model.add(Dense(64,activation=LeakyReLU(0.75)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.75)))
model.add(Dropout(1/32))
model.add(Dense(32,activation=LeakyReLU(0.75)))
model.add(Dropout(1/16))
model.add(Dense(y_train.shape[1],activation='softmax'))

# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# model.fit(xy_train[0][0],xy_train[0][1],epochs=10)
hist=model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),batch_size=4,
                    callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=100,restore_best_weights=True,verbose=True))

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
'''