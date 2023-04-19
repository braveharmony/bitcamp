import numpy as np
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict\
    ,GridSearchCV
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,load_digits
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,name):
    from time import time
    startTime=time()
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

    kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

    def mergedict(a,b):
        parameterlists = [{"n_estimators":[100,200]}]
        parameterlists.append({'max_depth':[6,8,10,12]})
        parameterlists.append({'min_samples_leaf':[3,5,7,10]})
        parameterlists.append({'min_samples_split':[2,3,5,10]})
        parameterlists.append({'n_jobs':[-1,2,4]})
        return {**parameterlists[a],**parameterlists[b]}
    parameters=[]
    parameters.append(mergedict(0,2))
    parameters.append(mergedict(1,2))
    parameters.append(mergedict(2,3))
    parameters.append(mergedict(3,4))
    parameters.append(mergedict(1,4))
    model=GridSearchCV(RandomForestClassifier(),parameters,cv=kfold,verbose=1,
                    #    refit=False,
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
for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
    x,y=i(return_X_y=True)
    run_model(x,y,i.__name__)
datasets=pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0)
x=datasets.drop(datasets.columns[-1],axis=1)
y=datasets[datasets.columns[-1]]
run_model(x,y,'dacon_diabete')