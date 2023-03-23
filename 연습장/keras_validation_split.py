# 0. seed initialization
import numpy as np
import tensorflow as tf
import random
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9)

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model=Sequential(Dense(1,input_dim=x_train.shape[1]))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=len(x_train),epochs=10
          ,validation_split=0.2,verbose=True)