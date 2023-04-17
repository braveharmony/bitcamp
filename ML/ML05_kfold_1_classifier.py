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
from sklearn.utils import all_estimators
import warnings
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler

warnings.filterwarnings('ignore')

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
        data=f'{name}'
    else:
        x,y=datasets(return_X_y=True)
        data=datasets.__name__
    max_name=str()
    max_acc=0
    for index,model in all_estimators(type_filter='classifier'):
        try:
            model=model()
            kfold=KFold(n_splits=5,shuffle=True)
            scores=cross_val_score(model,x,y,cv=kfold)
            print(f'index : {index}')
            if isinstance(datasets, pd.DataFrame):
                print(f'{name} score : {scores}')
            else:
                print(f'{datasets.__name__} score : {scores}')
            if np.mean(scores)>max_acc:
                max_acc=np.mean(scores)
                max_name=index
        except Exception as e : print(f'{i}번 index : {index}\n실패\nerror:{e}')
        print('=====================================================')
    return f'data : {data} model:{max_name} acc:{max_acc}\n'
    
ans=str()
for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
    ans+=Classify_kfold(datasets=i)
ans+=Classify_kfold(pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0),'dacon_diabete')
print(ans)

# data : load_iris model:LinearDiscriminantAnalysis acc:0.9800000000000001
# data : load_breast_cancer model:GradientBoostingClassifier acc:0.9596335972675052
# data : load_wine model:QuadraticDiscriminantAnalysis acc:0.9944444444444445
# data : load_digits model:KNeighborsClassifier acc:0.9877592076756423
# data : dacon_diabete model:LinearDiscriminantAnalysis acc:0.7761244862008221