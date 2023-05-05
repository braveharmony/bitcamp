import numpy as np
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict\
    ,RandomizedSearchCV
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,load_digits,fetch_covtype
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from xgboost import XGBClassifier
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,name):
    from time import time
    startTime=time()
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

    kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)


    parameters=[
    {'n_estimators':[100,200,300],'learning_rate':[0.1,0.3,0.001,0.01],'max_depth':[4,5,6]},
    {'n_estimators':[90,100,110],'learning_rate':[0.1,0.001,0.01],'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1]},
    {'n_estimators':[90,110],'learning_rate':[0.1,0.001,0.5],'max_depth':[4,5,6],'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.7,0.8]}
    ]
    model=RandomizedSearchCV(XGBClassifier(
                    tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0),parameters,cv=kfold,verbose=1,
                    #    refit=False,
                    n_iter=5,
                    n_jobs=-1)

    model.fit(x_train,y_train)

    print(f'====================================={name}=====================================')
    print(f'최적의 매개변수 : {model.best_params_}\n최적의 파라미터 : {model.best_estimator_}\nbest score : {model.best_score_}\nmodel score : {model.score(x_test,y_test)}')
    print(f'acc : {accuracy_score(y_test,model.predict(x_test))}')
    print(f'best acc : {accuracy_score(y_test,model.best_estimator_.predict(x_test))}')
    print(f'걸린 시간 : {round(time()-startTime,2)}초')

    import pandas as pd

    # print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True))
    # print(pd.DataFrame(model.cv_results_).columns)
    # path='./temp/'
    # pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True).to_csv(path+f'{name}_m10Gridsearch3.csv')
from sklearn.decomposition import PCA
for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
    x,y=i(return_X_y=True)
    pca=PCA(n_components=x.shape[1]//2)
    run_model(x,y,i.__name__)
datasets=pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0)
x=datasets.drop(datasets.columns[-1],axis=1)
y=datasets[datasets.columns[-1]]
pca=PCA(n_components=x.shape[1]//2)
run_model(x,y,'dacon_diabete')