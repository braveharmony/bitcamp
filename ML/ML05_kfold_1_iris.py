import tensorflow as tf
import numpy as np
import random
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_digits
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 1. data prepare
# 분류
# 01_iris
# 02_cancer
# 03_dacon_diabets
# 04_wine
# 05_fetch_covtype
# 06_digit

# 08_california
# 09_diabete
# 10_ddarung
# 11_kaggle_bike

def Classify_kfold(datasets,name=''):
    if isinstance(datasets, pd.DataFrame):
        datasets=datasets.dropna()
        x=datasets.drop(datasets.columns[-1],axis=1)
        y=datasets[datasets.columns[-1]]
    else:
        x,y=datasets(return_X_y=True)
    kfold=KFold(n_splits=5,shuffle=True)
    model=RandomForestClassifier()
    scores=cross_val_score(model,x,y,cv=kfold)
    if isinstance(datasets, pd.DataFrame):
        print(f'{name} score : {scores}')
    else:
        print(f'{datasets.__name__} score : {scores}')

for i in [load_iris,load_breast_cancer,load_wine,load_digits,fetch_covtype]:
    Classify_kfold(datasets=i)
Classify_kfold(pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0),'dacon_diabete')

def Regressor_kfold(datasets,name=''):
    if isinstance(datasets, pd.DataFrame):
        datasets=datasets.dropna()
        x=datasets.drop(datasets.columns[-1],axis=1)
        y=datasets[datasets.columns[-1]]
    else:
        x,y=datasets(return_X_y=True)
    kfold=KFold(n_splits=5,shuffle=True)
    model=RandomForestRegressor()
    scores=cross_val_score(model,x,y,cv=kfold)
    if isinstance(datasets, pd.DataFrame):
        print(f'{name} score : {scores}')
    else:
        print(f'{datasets.__name__} score : {scores}')
from sklearn.datasets import fetch_california_housing,load_diabetes
for i in [fetch_california_housing,load_diabetes]:
    Regressor_kfold(datasets=i)
Regressor_kfold(pd.read_csv('./_data/DDarung/train.csv',index_col=0),'DDarung')
Regressor_kfold(pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0),'kaggle_bike')
