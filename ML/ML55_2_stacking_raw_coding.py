from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
from typing import List
x,y=load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

class stacking_model:
    def __init__(self):
        self.models:List[BaseEstimator]=[XGBClassifier(),CatBoostClassifier(verbose=False),LGBMClassifier()]
        self.final_model=CatBoostClassifier(verbose=False)
    
    def fit(self,x_train,y_train,**args):
        h_pre_for_train=[]
        for model in self.models:
            model.fit(x_train,y_train,**args)
            h_pre_for_train.append(model.predict(x_train))
        h_train=np.array(h_pre_for_train).T
        self.final_model.fit(h_train,y_train,**args)
    
    def predict(self,x_test,**args)->np.ndarray:
        h_predict=[]
        for model in self.models:
            h_predict.append(model.predict(x_test,**args))
        h_predict=np.array(h_predict).T
        return self.final_model.predict(h_predict,**args)
    
    def score(self,x_test,y_test,**args)->float:
        y_pred=self.predict(x_test,**args)
        return accuracy_score(y_test,y_pred)
    

model=stacking_model()

model.fit(x_train,y_train)

print(model.score(x_test,y_test))