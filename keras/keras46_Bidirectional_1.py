import numpy as np
import tensorflow as tf
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,LSTM,GRU,Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1.data prepare
datasets = np.array([i for i in range(1,10)])
timestep=3
def split_to_time(dataset,timstep):
    gen=(dataset[i:i+timestep] for i in range(len(dataset)-timestep+1))
    return np.array(list(gen))
x=split_to_time(datasets,timestep)
y=x[:,-1]+1
print(x.shape,y.shape)
x=np.reshape(x,list(x.shape)+[1])

# 2. model build
model=Sequential()
model.add(Bidirectional(SimpleRNN(10),input_shape=x.shape[1:]))
model.add(Dense(1))
model.summary()