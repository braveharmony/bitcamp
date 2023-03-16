import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D,Dense,Flatten,Dropout,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# 0. seed initialization

# 1. data.prepare
(x_train, y_train),(x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape,y_test.shape)
print(np.unique(x_train))
print(np.unique(y_train))

plt.imshow(x_train[0],'gray')
plt.show()