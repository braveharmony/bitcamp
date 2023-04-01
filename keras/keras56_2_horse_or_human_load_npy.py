# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import time


# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. x prepare

save_start=time.time()
datagen=ImageDataGenerator(
    rescale=1/255,
)

path='d:/study_data/_data/horse-or-human/train'

xy=datagen.flow_from_directory(directory=path
                                  ,target_size=(100,100)
                                  ,batch_size=24998
                                  ,class_mode='categorical'
                                #   ,color_mode='grayscale'
                                  ,color_mode='rgb'
                                  ,shuffle=True
                                  )


# x=(xy[i][0] for i in range(len(xy)))
# y=(xy[i][1] for i in range(len(xy)))
# x=np.array(list(x))
# print(x.shape)
# x=np.reshape(x,([x.shape[0]*x.shape[1]]+list(x.shape[2:])))
# y=np.array(list(y))
# y=np.reshape(y,([y.shape[0]*y.shape[1]]+list(y.shape[2:])))

x=xy[0][0]
y=xy[0][1]
print(x.shape)
print(y.shape)
print(x[:5])
print(y[:5])

print(f'runtime for generate : {time.time()-save_start}')

save_start=time.time()
path='d:/study_data/_save/horse-or-human/'
np.save(file=f'{path}x.npy',arr=x)
np.save(file=f'{path}y.npy',arr=y)


print(f'runtime for save : {time.time()-save_start}')



save_start=time.time()
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

input_shape=x.shape[1:]

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D,Flatten,MaxPooling2D,Dense,Dropout,Input,LeakyReLU
model=Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128,kernel_size=(2,2),padding='valid',activation=LeakyReLU(0.75)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='same',activation=LeakyReLU(0.75)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(filters=128,kernel_size=(3,3),padding='valid',activation=LeakyReLU(0.75)))
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
model.summary()

# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
# model.fit(xy_train[0][0],xy_train[0][1],epochs=10)
hist=model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),batch_size=50,
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
