import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings(action='ignore')
# 1. data
data_list = [load_iris,load_breast_cancer,load_wine]
model_list=[LinearSVC(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]

for i in data_list:
    x,y=i(return_X_y=True)
    for model in model_list:
        model.fit(x,y)
        print('\n=============================================')
        print(f'model : {type(model).__name__} data: {i.__name__}')
        try:
            print(model.score(x,y))
        except:
            print('실행 실패')
        