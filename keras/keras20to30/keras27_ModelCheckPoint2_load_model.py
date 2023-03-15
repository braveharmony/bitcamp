import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Input,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. model build,compile,training
model=load_model('./_save/MCP/keras27_ModelCheckPoint.hdf5')

# 3. predict,evaluate
print(f'loss :  {model.evaluate(x_test,y_test)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')