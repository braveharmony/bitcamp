import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_wine
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import accuracy_score,r2_score
warnings.filterwarnings(action='ignore')
# 1. data
data_list = [load_iris,load_breast_cancer,load_wine]
model_list=[LinearSVC(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]

# 2. model
for i in data_list:
    x,y=i(return_X_y=True)
    for model in model_list:
        model.fit(x,y)
        print('\n=============================================')
        print(f'model name : {type(model).__name__}\ndata: {i.__name__}')
        try:
            print(f'model_score : {model.score(x,y)}')
            y_pred=model.predict(x)
            print(f'acc : {accuracy_score(y_pred,y)}')
            print(f'r2_score : {r2_score(y_pred,y)}')
        except:
            print('실행 실패')
        