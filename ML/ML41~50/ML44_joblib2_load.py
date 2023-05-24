from sklearn.datasets import load_breast_cancer,load_diabetes,fetch_california_housing,load_digits
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from typing import Tuple,Dict
import joblib
import os


x,y=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8,stratify=y)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


path='./_save/joblib_test/'
os.makedirs(path,exist_ok=True)
model:XGBClassifier=joblib.load(path+'ML43_joblib1_save.dat')
print(model.score(x_test,y_test))