import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,MaxPool2D,Conv2D,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from tensorflow.keras.datasets import cifar10
from matplotlib import pyplot as plt
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar10.load_data()
for i in range(20):
    plt.figure(i)
    plt.imshow(x_train[i])
    plt.title(y_train[i])
plt.show()