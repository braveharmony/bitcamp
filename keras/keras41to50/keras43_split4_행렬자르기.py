import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

datasets = np.array(range(1,41)).reshape(10,4)

x_data=datasets[:,:-1]
y_data=datasets[:,-1]
print(x_data.shape,y_data.shape)

ts=3

def split(x,y,ts):
    genx=(x[i:i+ts]for i in range(len(x)-ts))
    geny=(y[i+ts]for i in range(len(y)-ts))
    return np.array(list(genx)),np.array(y[ts:])#np.array(list(geny))

x,y=split(x_data,y_data,ts)
print(datasets)
print(x)
print(y)