from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
# 1. dataprepare
(x_train,y_train),(x_test,y_test)=reuters.load_data(num_words=10000,test_split=0.2)

from tensorflow.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train,padding='pre',truncating='pre',maxlen=100)
x_test = pad_sequences(x_test,padding='pre',truncating='pre',maxlen=100)

def onehot(y_train,y_test):
    y=np.concatenate((y_train,y_test))
    y=pd.get_dummies(y,prefix='number')
    return y[:len(y_train)],y[len(y_train):]
y_train,y_test=onehot(y_train,y_test)


# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input,Conv1D,Embedding,Dropout,Flatten,Dense,LeakyReLU,SimpleRNN
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Embedding(10000,32))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(Conv1D(64,15,activation=LeakyReLU(0.75)))
model.add(SimpleRNN(64,activation=LeakyReLU(0.75)))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(y_train.shape[1],activation='softmax'))
model.summary()

# 3. compile, training
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,validation_data=(x_test,y_test)
          ,batch_size=100,verbose=True,epochs=1000
          ,callbacks=EarlyStopping(monitor='val_acc',mode='max',patience=50))