import tensorflow as tf
import numpy as np
import random 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.datasets import load_boston
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib as mpl
import matplotlib.pyplot as plt

datasets=load_boston()
x=datasets.data
y=datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=1,shuffle=True)


model=Sequential()
model.add(Dense(16,input_dim=x.shape[1],activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='linear'))

model.compile(loss="mse",optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min',verbose=True,restore_best_weights=True)
hist=model.fit(x_train,y_train,verbose=True,batch_size=len(x),epochs=1000,validation_split=0.2,callbacks=es)

print(hist.history['val_loss'])
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'r2score : {r2_score(y_test,y_predict)}')

plt.plot(hist.history['loss'],marker='.',color='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',color='blue',label='val_loss')
plt.legend()
plt.grid()
plt.show()
