import numpy as np
import random
from sklearn.model_selection import train_test_split,KFold,cross_val_score,cross_val_predict\
    ,RandomizedSearchCV
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.metrics import r2_score
from collections.abc import Iterable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,name):
    from time import time
    startTime=time()
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)

    kfold=KFold(n_splits=3,shuffle=True,random_state=seed)

    def mergedict(a,b):
        parameterlists = [{"n_estimators":[100,200]}]
        parameterlists.append({'max_depth':[6,8]})
        parameterlists.append({'min_samples_leaf':[7,10]})
        parameterlists.append({'min_samples_split':[3,5,10]})
        return {**parameterlists[a],**parameterlists[b]}
    parameters=[]
    parameters.append(mergedict(0,2))
    parameters.append(mergedict(1,2))
    parameters.append(mergedict(2,3))
    model=RandomizedSearchCV(RandomForestRegressor(),parameters,cv=kfold,verbose=1,
                    #    refit=False,
                    n_jobs=-1)

    model.fit(x_train,y_train)

    print(f'====================================={name}=====================================')
    print(f'최적의 매개변수 : {model.best_params_}\n최적의 파라미터 : {model.best_estimator_}\nbest score : {model.best_score_}\nmodel score : {model.score(x_test,y_test)}')
    print(f'acc : {r2_score(y_test,model.predict(x_test))}')
    print(f'best acc : {r2_score(y_test,model.best_estimator_.predict(x_test))}')
    print(f'걸린 시간 : {round(time()-startTime,2)}초')

    import pandas as pd

    # print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True))
    # print(pd.DataFrame(model.cv_results_).columns)
    # path='./temp/'
    # pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True).to_csv(path+f'{name}_m10Gridsearch3.csv')
for i in [load_diabetes,fetch_california_housing]:
    x,y=i(return_X_y=True)
    run_model(x,y,i.__name__)
datasets=pd.read_csv('./_data/DDarung/train.csv',index_col=0)
datasets=datasets.dropna(
    
)
x=datasets.drop(datasets.columns[-1],axis=1)
y=datasets[datasets.columns[-1]]
run_model(x,y,'DDarung')
datasets=pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0)
x=datasets.drop(datasets.columns[-1],axis=1)
y=datasets[datasets.columns[-1]]
run_model(x,y,'kaggle_bike')
