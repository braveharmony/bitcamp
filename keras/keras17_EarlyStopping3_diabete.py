import tensorflow as tf
import numpy as np
import random
import pandas as pd
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from sklearn.datasets import load_diabetes
from matplotlib import pyplot as plt

seed=20580
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


datasets=load_diabetes()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed,shuffle=True)


model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))


es=EarlyStopping(monitor='val_loss',patience=10,verbose=True,mode='min',restore_best_weights=True)
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=len(x),verbose=True,validation_split=0.2,callbacks=es,epochs=100)


print(f'loss : {model.evaluate(x_test,y_test)}')
y_predict=model.predict(x_test)
print(f"r2score : {r2_score(y_test,y_predict)}")
