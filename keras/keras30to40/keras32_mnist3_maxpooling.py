import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time, datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Conv2D,MaxPool2D,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import mnist

# 1. data prepare
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test=x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
y_train=np.array(pd.get_dummies(y_train,prefix='number'))
y_test=np.array(pd.get_dummies(y_test,prefix='number'))

# 2. 모델
model=Sequential()
model.add(Conv2D(filters=8
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,input_size=(x_train.shape[0],x_train.shape[1],x_train.shape[2])
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=16
                 ,kernel_size=(3,3)
                 ,padding='same'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters=64
                 ,kernel_size=(4,4)
                 ,padding='valid'
                 ,activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
