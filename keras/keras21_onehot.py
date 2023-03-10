# [과제]
# 3가지 원핫 인코딩 방식을 비교할것
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data perpare
datasets=load_iris()
x=datasets.data
y=datasets.target+1
print(f'========================== origin ===========================')
print(f'y.shape : {y.shape} yclass : {y.__class__} ylabel : {np.unique(np.array(y))}')
print(f'========================== origin ===========================')

# 1. pandas의 get_dummies
y=datasets.target+1
y=pd.get_dummies(y,prefix='number')
print(f'========================== pandas ===========================')
print(f'y.shape : {y.shape} yclass : {y.__class__} ylabel : {np.unique(np.array(y))}')
print(f'========================== pandas ===========================')

# 2. keras의 to_categorical
from keras.utils import to_categorical
y=datasets.target+1
y=to_categorical(y)
print(f'========================= sklearn ===========================')
print(f'y.shape : {y.shape} yclass : {y.__class__} ylabel : {np.unique(np.array(y))}')
print(f'========================= sklearn ===========================')

# 3. sklearn의 OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
y=datasets.target+1
y=np.array([y]).T
y=OneHotEncoder().fit_transform(y).toarray()
print(f'========================== keras ============================')
print(f'y.shape : {y.shape} yclass : {y.__class__} ylabel : {np.unique(np.array(y))}')
print(f'========================== keras ============================')


# 미세한 차이를 정리하시오