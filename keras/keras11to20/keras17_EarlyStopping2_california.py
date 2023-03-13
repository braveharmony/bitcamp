import tensorflow as tf
import numpy as np
import random
import matplotlib as mpl
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.datasets import fetch_california_housing
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

#1. data prepare
datasets=fetch_california_housing()

x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)

#2. model building
model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))

#3. compile, training
es=EarlyStopping(patience=5,verbose=True,monitor='val_loss',mode="min",restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
model.fit(x,y,batch_size=len(x_train)//4,verbose=True,validation_split=0.2,epochs=1000,callbacks=es)

#4. evaluate,predict
loss=model.evaluate(x_test,y_test,batch_size=len(x_test)//10)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'r2score : {r2_score(y_test,y_predict)}')
