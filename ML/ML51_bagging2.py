import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
# x,y=load_breast_cancer(return_X_y=True)

# x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8,shuffle=True,stratify=y)

# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

# 2. model build
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn. ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.datasets import load_breast_cancer,load_digits,load_iris,fetch_covtype

def run_model(dataset:load_breast_cancer,model:DecisionTreeClassifier)->DecisionTreeClassifier:
    x,y=dataset(return_X_y=True)
    
    x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8,shuffle=True,stratify=y)
    
    scaler=StandardScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    if model==BaggingClassifier:
        model=BaggingClassifier(DecisionTreeClassifier(),200,n_jobs=-1,random_state=0,bootstrap=True)
    else:
        model=model()
    model.fit(x_train,y_train)
    print(f'dataset : {dataset.__name__} model : {type(model).__name__}\nscore : {model.score(x_test,y_test)}')
    return model

model=DecisionTreeClassifier()

datasets=[load_breast_cancer,load_digits,load_iris,fetch_covtype]
models=[DecisionTreeClassifier,RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,XGBClassifier]
for i in datasets:
    for j in models:
        run_model(i,j)
