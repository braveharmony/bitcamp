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
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.utils import all_estimators
import warnings
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler
warnings.filterwarnings('ignore')

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def Regressor_kfold(datasets,name='',scaler=MinMaxScaler):
    if isinstance(datasets, pd.DataFrame):
        datasets=datasets.dropna()
        x=datasets.drop(datasets.columns[-1],axis=1)
        y=datasets[datasets.columns[-1]]
        data=f'{name}'
    else:
        x,y=datasets(return_X_y=True)
        data=f'{datasets.__name__}'
    scaler=scaler()
    scaler_name=type(scaler).__name__
    x=scaler.fit_transform(x)
    max_name=str()
    max_acc=0
    for index,model in all_estimators(type_filter='regressor'):
        if index=='GaussianProcessRegressor':
            continue
        try:
            model=model()
            kfold=KFold(n_splits=5,shuffle=True)
            scores=cross_val_score(model,x,y,cv=kfold)
            print(f'index : {index} scaler : {scaler_name}')
            if isinstance(datasets, pd.DataFrame):
                print(f'{name} score : {scores}')
            else:
                print(f'{datasets.__name__} score : {scores}')
            if np.mean(scores)>max_acc:
                max_acc=np.mean(scores)
                max_name=index
        except Exception as e : print(f'{i}번 index : {index}\n실패\nerror:{e}')
        print('=====================================================')
    return f'data : {data} scaler: {scaler_name} model:{max_name} acc:{max_acc}\n'
    


ans=str()
for scaler in [MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler]:
    for i in [fetch_california_housing,load_diabetes]:
        ans+=Regressor_kfold(datasets=i)
    ans+=Regressor_kfold(pd.read_csv('./_data/DDarung/train.csv',index_col=0),'DDarung')
    kaggle_bike=pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0)
    ans+=Regressor_kfold(kaggle_bike.drop([kaggle_bike.columns[-3],kaggle_bike.columns[-2]],axis=1),'kaggle_bike')
print(ans)

# GaussianProcessRegressor 더럽게 느림,결과도 개구데기