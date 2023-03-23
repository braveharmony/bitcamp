# 0. seed initializiation
import numpy as np
import tensorflow as tf
import random
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.datasets import load_breast_cancer

datasets = load_breast_cancer()
print(datasets.DESCR)


print(datasets.target_names)
print(np.unique(datasets.target))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(datasets.data
                                               ,datasets.target,train_size=0.8)

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model=Sequential(Dense(1,input_shape=x_train.shape[1:],activation='sigmoid'))

# 3. compile
model.compile(loss='binary_crossentropy',optimizer='adam')

