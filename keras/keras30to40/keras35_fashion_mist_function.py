from tensorflow.keras.datasets import fashion_mnist
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time, datetime
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPool2D,Dropout,Input,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. 모델
input=Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))
layer=Conv2D(filters=64
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu')(input)
layer=MaxPool2D(pool_size=(2,2))(layer)
layer=Conv2D(filters=128
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu')(layer)
layer=MaxPool2D(pool_size=(2,2))(layer)
layer=Conv2D(filters=64
                 ,kernel_size=(4,4)
                 ,padding='valid'
                 ,activation='relu')(layer)
layer=MaxPool2D(pool_size=(2,2))(layer)
layer=Conv2D(filters=128
                 ,kernel_size=(2,2)
                 ,padding='same'
                 ,activation='relu')(layer)
layer=MaxPool2D(2,2)(layer)
layer=Flatten()(layer)
layer=Dense(32,activation='relu')(layer)
layer=Dropout(0.125)(layer)
layer=Dense(32,activation='relu')(layer)
layer=Dropout(0.125)(layer)
layer=Dense(32,activation='relu')(layer)
layer=Dropout(0.125)(layer)
layer=Dense(10,activation='softmax')(layer)
model=Model(inputs=input,outputs=layer)
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
date=datetime.datetime.now()
date = date.strftime('%H%M')
model.save(f'./_save/keras32/{date}.h5')