import numpy as np
import random
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict\
    ,RandomizedSearchCV,HalvingGridSearchCV,HalvingRandomSearchCV
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,load_digits,fetch_covtype
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
    print(f'====================================={name}=====================================')
    from time import time
    startTime=time()
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

    kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

    def mergedict(a,b):
        parameterlists = [{"n_estimators":[100,200]}]
        parameterlists.append({'max_depth':[6,8,10,12]})
        parameterlists.append({'min_samples_leaf':[3,5,7,10]})
        parameterlists.append({'min_samples_split':[2,3,5,10]})
        # parameterlists.append({'n_jobs':[-1,2,4]})
        return {**parameterlists[a],**parameterlists[b]}
    parameters=[]
    parameters.append(mergedict(0,2))
    parameters.append(mergedict(1,2))
    parameters.append(mergedict(2,3))
    
    parameters.append(mergedict(0,3))
    
    # parameters.append(mergedict(3,4))
    # parameters.append(mergedict(1,4))
    model=HalvingRandomSearchCV(RandomForestClassifier(),parameters,cv=kfold,verbose=1,
                    # refit=False,
                    # n_iter=5,
                    factor= np.exp(1),
                    n_jobs=-1)

    model.fit(x_train,y_train)

    # print(f'최적의 매개변수 : {model.best_params_}\n최적의 파라미터 : {model.best_estimator_}\nbest score : {model.best_score_}\nmodel score : {model.score(x_test,y_test)}')
    print(f'acc : {accuracy_score(y_test,model.predict(x_test))}')
    # print(f'best acc : {accuracy_score(y_test,model.best_estimator_.predict(x_test))}')
    print(f'걸린 시간 : {round(time()-startTime,2)}초')
    
    import pandas as pd

    # print(pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True))
    # print(pd.DataFrame(model.cv_results_).columns)
    # path='./temp/'
    # pd.DataFrame(model.cv_results_).sort_values('rank_test_score',ascending=True).to_csv(path+f'{name}_m10Gridsearch3.csv')
for i in [load_iris,load_breast_cancer,load_wine,\
    load_digits]:
    x,y=i(return_X_y=True)
    run_model(x,y,i.__name__)
datasets=pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0)
x=datasets.drop(datasets.columns[-1],axis=1)
y=datasets[datasets.columns[-1]]
run_model(x,y,'dacon_diabete')
# =====================================load_iris=====================================
# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 120
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 4
# n_resources: 30
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 1
# n_candidates: 2
# n_resources: 81
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# Traceback (most recent call last):
#   File "c:\study\ML\ML15_halvinrandomSearch1.py", line 63, in <module>
#     run_model(x,y,i.__name__)
#   File "c:\study\ML\ML15_halvinrandomSearch1.py", line 51, in run_model
#     print(f'best acc : {accuracy_score(y_test,model.best_estimator_.predict(x_test))}')
# AttributeError: 'HalvingRandomSearchCV' object has no attribute 'best_estimator_'

# (tf274gpu) C:\study> c: && cd c:\study && cmd /C "C:\Users\bitcamp\anaconda3\envs\tf274gpu\python.exe c:\Users\bitcamp\.vscode\extensions\ms-python.python-2023.6.0\pythonFiles\lib\python\debugpy\adapter/../..\debugpy\launcher 58665 -- c:\study\ML\ML15_halvinrandomSearch1.py "
# =====================================load_iris=====================================
# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 120
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 4
# n_resources: 30
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 1
# n_candidates: 2
# n_resources: 81
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc : 0.9666666666666667
# 걸린 시간 : 3.87초
# =====================================load_breast_cancer=====================================
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 20
# max_resources_: 455
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 22
# n_resources: 20
# Fitting 5 folds for each of 22 candidates, totalling 110 fits
# ----------
# iter: 1
# n_candidates: 9
# n_resources: 54
# Fitting 5 folds for each of 9 candidates, totalling 45 fits
# ----------
# iter: 2
# n_candidates: 4
# n_resources: 147
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 401
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc : 0.9473684210526315
# 걸린 시간 : 10.1초
# =====================================load_wine=====================================
# n_iterations: 2
# n_required_iterations: 2
# n_possible_iterations: 2
# min_resources_: 30
# max_resources_: 142
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 4
# n_resources: 30
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 1
# n_candidates: 2
# n_resources: 81
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc : 1.0
# 걸린 시간 : 1.56초
# =====================================load_digits=====================================
# n_iterations: 3
# n_required_iterations: 3
# n_possible_iterations: 3
# min_resources_: 100
# max_resources_: 1437
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 14
# n_resources: 100
# Fitting 5 folds for each of 14 candidates, totalling 70 fits
# ----------
# iter: 1
# n_candidates: 6
# n_resources: 271
# Fitting 5 folds for each of 6 candidates, totalling 30 fits
# ----------
# iter: 2
# n_candidates: 3
# n_resources: 738
# Fitting 5 folds for each of 3 candidates, totalling 15 fits
# acc : 0.9694444444444444
# 걸린 시간 : 6.96초
# =====================================dacon_diabete=====================================
# n_iterations: 4
# n_required_iterations: 4
# n_possible_iterations: 4
# min_resources_: 20
# max_resources_: 521
# aggressive_elimination: False
# factor: 2.718281828459045
# ----------
# iter: 0
# n_candidates: 26
# n_resources: 20
# Fitting 5 folds for each of 26 candidates, totalling 130 fits
# ----------
# iter: 1
# n_candidates: 10
# n_resources: 54
# Fitting 5 folds for each of 10 candidates, totalling 50 fits
# ----------
# iter: 2
# n_candidates: 4
# n_resources: 147
# Fitting 5 folds for each of 4 candidates, totalling 20 fits
# ----------
# iter: 3
# n_candidates: 2
# n_resources: 401
# Fitting 5 folds for each of 2 candidates, totalling 10 fits
# acc : 0.732824427480916
# 걸린 시간 : 11.09초

