import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score,cross_val_predict\
    ,train_test_split,KFold
from sklearn.metrics import accuracy_score
import random
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,stratify=y,random_state=seed)

n_split=5
kfold=KFold(n_splits=n_split,shuffle=True,random_state=seed)

# 2. model build
model=SVC()
# model=RandomForestClassifier()

score=cross_val_score(model,x_train,y_train,cv=kfold)
print(f'cross_val_score : {score} \nmean_csv: {round(np.mean(score),4)}')

y_predict=cross_val_predict(model,x_test,y_test,cv=kfold)
acc=f'cross_val_predict ACC : {accuracy_score(y_test,y_predict)}'
print(acc)