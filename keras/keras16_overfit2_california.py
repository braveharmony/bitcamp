import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=fetch_california_housing()
# print(datasets.DESCR)
x=datasets['data']
y=datasets['target']
feature_names=datasets['feature_names']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)

# model build
model=Sequential()
model.add(Dense(10,input_dim=x.shape[1],activation="relu"))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

# compile,training
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x,y,batch_size=1000,validation_split=0.2,verbose=True,epochs=20)

# evaluate, predict
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'r2 : {r2_score(y_test,y_predict)}')


# ploting
plt.plot(hist.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.grid()
plt.show()