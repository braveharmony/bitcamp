import numpy as np
import random
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,cross_val_predict
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

gamma=[0.001,0.01,0.1,1,10,100]
C=[0.001,0.01,0.1,1,10,100]

def run_model(model_type,gamma,C,cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=0)):
    max_gamma=0
    max_c=0
    max_score=0
    for i in gamma:
        for j in C:
            model=model_type(gamma=i,C=j)
            cross_val_score(model,x_train,x_test,cv=cv)
            score=accuracy_score(y_test,cross_val_predict(model,x_test,y_test,cv=cv))
            print(f"acc : {score}")
            if max_score<score:
                best_parameter={'gamma':i,'C':j}
                max_score=score
    return f"max_gamma : {best_parameter['gamma']} max_c : {best_parameter['C']} max_score : {max_score}"

print(run_model(SVC,gamma,C,cv=kfold))