import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')

print(df.columns,dft.columns)
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]

