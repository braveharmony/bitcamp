import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import pandas as pd

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
x_train=np.array(range(1,17))
y_train=np.array(range(1,17))

x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,train_size=13/16+0.0001,random_state=seed)
x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,train_size=10/13+0.0001,random_state=seed)

print(x_train.shape,x_val.shape,x_test.shape)

# 2. model build
model=Sequential()
model.add(Dense(5,input_dim=1))
model.add(Dense(1))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=len(x_train),epochs=3000,validation_data=(x_val,y_val))

# 4. evaluate,predict
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'r2 : {r2_score(y_test,y_predict)}')