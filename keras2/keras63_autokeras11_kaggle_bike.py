import autokeras as ak
from keras.datasets import mnist
from typing import Tuple
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_diabetes
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


import pandas as pd
path='./_data/kaggle_bike/'
datasets=pd.read_csv(path+'train.csv',index_col=0).dropna()
x=datasets.drop(datasets.columns[-3:-1],axis=1)
y=datasets[datasets.columns[-1]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

model=ak.StructuredDataRegressor(max_trials=1)
model.fit(x_train,y_train,epochs=1,batch_size=100,validation_data=(x_test,y_test))