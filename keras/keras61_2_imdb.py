from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import tensorflow as tf
import random
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 1. data prepare
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_words=10000)

# print(np.unique(y_train,return_counts=True))
# print(pd.value_counts(y_train))
# print(x_train.shape,y_train.shape)
# print(x_test.shape,y_test.shape)
# print('영화평의 최대 길이 : ',max(map(len,x_train)))
# print('영화평의 평균 길이 : ',sum(map(len,x_train))/len(x_train))
from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train=pad_sequences(x_train,maxlen=240,padding='pre',truncating='pre')
x_test=pad_sequences(x_test,maxlen=240,padding='pre',truncating='pre')
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)

# x_train=np.zeros((1,240))

# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,Conv1D,SimpleRNN,Dropout,Input,LeakyReLU
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Embedding(10000,128))
for i in range(7):
    model.add(Conv1D(32,31,activation=LeakyReLU(0.5)))
model.add(SimpleRNN(32))
for i in range(15):
    model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dense(1,activation='sigmoid'))
model.summary()

# 3.compile, training
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,validation_data=(x_test,y_test)
          ,batch_size=1000,epochs=100,verbose=True
          ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=3,restore_best_weights=True,verbose=True))