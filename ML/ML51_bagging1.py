import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score

x,y=load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=0.8,shuffle=True,stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. model build
from sklearn.tree import DecisionTreeClassifier

model=DecisionTreeClassifier()

# 3. training
model.fit(x_train,y_train)

# 4. evaluate
print(model.score(x_test,y_test))

from sklearn.ensemble import RandomForestClassifier

model=RandomForestClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

from sklearn.ensemble import BaggingClassifier
model=BaggingClassifier(DecisionTreeClassifier(),200,n_jobs=-1,random_state=0,bootstrap=True)
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

from sklearn. ensemble import GradientBoostingClassifier
model=GradientBoostingClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))

from xgboost import XGBClassifier
model=XGBClassifier()
model.fit(x_train,y_train)
print(model.score(x_test,y_test))
