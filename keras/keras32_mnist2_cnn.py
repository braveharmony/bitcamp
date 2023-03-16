import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,LeakyReLU
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
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)#/255.
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)#/255.
y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. model build
model=Sequential()
model.add(Conv2D(filters=8
                ,kernel_size=(3,3)
                ,padding='same'
                ,input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                ,activation='relu'))
model.add(Conv2D(filters=16
                ,kernel_size=(3,3)
                ,padding='same'
                ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=32
                ,kernel_size=(4,4)
                ,padding='valid'
                ,activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=64
                ,kernel_size=(3,3)
                ,padding='same'
                ,activation='relu'))
model.add(MaxPool2D(2,2))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(10,activation='softmax'))
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
# 5. plotting
