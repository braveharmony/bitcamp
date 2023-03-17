import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,MaxPool2D,Conv2D,LeakyReLU,Flatten
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar100

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()
print(x_train.shape,x_test.shape)
print(y_train.shape,y_test.shape)


x_train=x_train/255.
x_test=x_test/255.
y_train=np.array(pd.get_dummies(y_train.T[0],prefix='number'))
y_test=np.array(pd.get_dummies(y_test.T[0],prefix='number'))

# 2. model build
model=Sequential()
model.add(Conv2D(filters=512
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'
                 ,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3],)))
model.add(Conv2D(filters=256
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=512
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=256
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=256
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=64
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=256
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=100,activation='softmax'))
model.summary()

# 3. compile,training
model.compile(loss='categorical_crossentropy',optizier='adam',metrics=['acc'])
model.fit(x_train,y_train,epochs=1
          ,batch_size=len(x_train)//200,validation_split=0.8
          ,callbacks=EarlyStopping(moniter='val_loss',mode='min',patience=20
                                   ,restore_best_weights=True,verbose=True)
          ,verbose=True)

# 4. predict,evaluate
date=datetime.datetime().strftime('%H시%M분')
acc=accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1))
model.save(f'./_save/keras34/acc_{acc}_time_{acc}.h5')