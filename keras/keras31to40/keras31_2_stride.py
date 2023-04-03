import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from matplotlib import pyplot as plt
from tensorflow.python.keras.models import Model,load_model,Sequential
from tensorflow.python.keras.layers import Dense,Input,LeakyReLU,Dropout,MaxPooling2D,Conv2D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 0. seed initialization
seed=0 
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare


# 2 . model build
model=Sequential()
model.add(Conv2D(filters=7,
                 kernel_size=(2,2),
                 padding='same',                
                 input_shape=(8,8,1)))          #출력 : (N, 7, 7, 7) 1*2*2*7+7
model.add(MaxPooling2D(pool_size=(2,2)))        # (batch_size, rows, colums, channels)
model.add(Conv2D(filters=4,
                 kernel_size=(3,3),
                 padding='same',
                 activation='relu'))            #출력 : (N, 5, 5, 4) 7*3*3*4+4
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10,(2,2),activation='relu'))   #출력 : (n, 4, 4, 10) 4*2*2*10+10
model.add(Flatten())                            
model.add(Dense(64,activation='relu'))          
model.add(Dense(64,activation='relu'))          
model.add(Dense(3,activation='softmax'))        
model.summary()                                 
