import numpy as np
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict\
    ,GridSearchCV
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from collections.abc import Iterable
from sklearn.svm import SVC

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

# 1. data prepare
x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

kfold=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed)

Parameters = [{"C":[1,10,100], "kernel" : ['rbf','linear','sigmoid'], 'degree':[3,4,5], 'gamma':[0.01,0.001, 0.0001]}] 
model=GridSearchCV(SVC(),Parameters,cv=kfold,verbose=1,refit=True,n_jobs=-1)

model.fit(x_train,y_train)

print(f'최적의 매개변수 : {model.best_params_}\n최적의 파라미터 : {model.best_estimator_}\nbest score : {model.best_score_}\nmodel score : {model.score(x_test,y_test)}')