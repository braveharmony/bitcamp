import tensorflow as tf
import numpy as np
import random
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,fetch_covtype,load_digits
from sklearn.model_selection import train_test_split,cross_val_score, KFold ,cross_val_predict
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.utils import all_estimators
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler
import warnings
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

def Classify_kfold(datasets,name='',scaler=MinMaxScaler):
    if isinstance(datasets, pd.DataFrame):
        datasets=datasets.dropna()
        x=datasets.drop(datasets.columns[-1],axis=1)
        y=datasets[datasets.columns[-1]]
        data=f'{name}'
    else:
        x,y=datasets(return_X_y=True)
        data=datasets.__name__
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y)
    scaler=scaler()
    scaler_name=type(scaler).__name__
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    max_name=str()
    max_acc=0
    
    for index,model in all_estimators(type_filter='classifier'):
        try:
            model=model()
            kfold=KFold(n_splits=5,shuffle=True)
            cross_val_score(model,x_train,y_train,cv=kfold)
            scores=accuracy_score(y_test,cross_val_predict(model,x_test,y_test,cv=kfold))
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
    for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
        ans+=Classify_kfold(datasets=i,scaler=scaler)
    ans+=Classify_kfold(pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0),'dacon_diabete',scaler=scaler)
print(ans)
# data : load_iris scaler: MinMaxScaler model:BaggingClassifier acc:1.0
# data : load_breast_cancer scaler: MinMaxScaler model:LabelPropagation acc:0.9912280701754386
# data : load_wine scaler: MinMaxScaler model:GaussianProcessClassifier acc:0.9722222222222222
# data : load_digits scaler: MinMaxScaler model:SVC acc:0.9583333333333334
# data : dacon_diabete scaler: MinMaxScaler model:ExtraTreesClassifier acc:0.7633587786259542
# data : load_iris scaler: MaxAbsScaler model:NuSVC acc:1.0
# data : load_breast_cancer scaler: MaxAbsScaler model:LabelSpreading acc:0.9912280701754386
# data : load_wine scaler: MaxAbsScaler model:ExtraTreesClassifier acc:1.0
# data : load_digits scaler: MaxAbsScaler model:GaussianProcessClassifier acc:0.9694444444444444
# data : dacon_diabete scaler: MaxAbsScaler model:CalibratedClassifierCV acc:0.7786259541984732
# data : load_iris scaler: StandardScaler model:LinearDiscriminantAnalysis acc:1.0
# data : load_breast_cancer scaler: StandardScaler model:GaussianProcessClassifier acc:0.9736842105263158
# data : load_wine scaler: StandardScaler model:LinearDiscriminantAnalysis acc:1.0
# data : load_digits scaler: StandardScaler model:LogisticRegression acc:0.9305555555555556
# data : dacon_diabete scaler: StandardScaler model:BernoulliNB acc:0.7709923664122137
# data : load_iris scaler: RobustScaler model:AdaBoostClassifier acc:1.0
# data : load_breast_cancer scaler: RobustScaler model:LinearSVC acc:0.956140350877193
# data : load_wine scaler: RobustScaler model:CalibratedClassifierCV acc:1.0
# data : load_digits scaler: RobustScaler model:ExtraTreesClassifier acc:0.95
# data : dacon_diabete scaler: RobustScaler model:LinearSVC acc:0.7938931297709924