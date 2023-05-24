import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_wine,load_digits,fetch_covtype
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

seed=0

dataset=fetch_covtype

x,y=dataset(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y,random_state=seed)


from lightgbm import LGBMClassifier
model=LGBMClassifier()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))

x=PolynomialFeatures(degree=2).fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y,random_state=seed)


from lightgbm import LGBMClassifier
model=LGBMClassifier()
model.fit(x_train,y_train)

print(model.score(x_test,y_test))