from sklearn.datasets import fetch_california_housing,load_iris
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler,RobustScaler
from sklearn.preprocessing import QuantileTransformer,PowerTransformer
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
import random
from typing import List,Tuple
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

# 1. data prepare
def scaling(x_train,x_test,scaler:MinMaxScaler)->Tuple[np.ndarray,np.ndarray]:
    scaler=scaler()    
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test

def scaling_with_method(x_train,x_test,scaler:MinMaxScaler,method:str='box-cox')->Tuple[np.ndarray,np.ndarray]:
    scaler=scaler(method=method)    
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    return x_train,x_test


x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,train_size=0.8,random_state=seed,stratify=y)


model=LogisticRegression()
print('========================================================')
print(f'model : {type(model).__name__}')


scaler=QuantileTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=StandardScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MinMaxScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MaxAbsScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=RobustScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='yeo-johnson')
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}yeo-johnson')
print(f'result : {model.score(x_test,y_test)}')

# scaler=PowerTransformer
# x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='box-cox')
# model.fit(x_train,y_train)
# print(f'scaler : {scaler.__name__}box-cox')
# print(f'result : {model.score(x_test,y_test)}')



model=RandomForestClassifier()
print('========================================================')
print(f'model : {type(model).__name__}')

scaler=QuantileTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=StandardScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MinMaxScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MaxAbsScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=RobustScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='yeo-johnson')
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}yeo-johnson')
print(f'result : {model.score(x_test,y_test)}')

# scaler=PowerTransformer
# x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='box-cox')
# model.fit(x_train,y_train)
# print(f'scaler : {scaler.__name__}box-cox')
# print(f'result : {model.score(x_test,y_test)}')






x,y=fetch_california_housing(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,train_size=0.8,random_state=seed)

model=LinearRegression()
print('========================================================')
print(f'model : {type(model).__name__}')

scaler=QuantileTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=StandardScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MinMaxScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MaxAbsScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=RobustScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='yeo-johnson')
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}yeo-johnson')
print(f'result : {model.score(x_test,y_test)}')

# scaler=PowerTransformer
# x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='box-cox')
# model.fit(x_train,y_train)
# print(f'scaler : {scaler.__name__}box-cox')
# print(f'result : {model.score(x_test,y_test)}')



model=RandomForestRegressor()
print('========================================================')
print(f'model : {type(model).__name__}')

scaler=QuantileTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=StandardScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MinMaxScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=MaxAbsScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=RobustScaler
x_train,x_test=scaling(x_train,x_test,scaler)
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}')
print(f'result : {model.score(x_test,y_test)}')

scaler=PowerTransformer
x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='yeo-johnson')
model.fit(x_train,y_train)
print(f'scaler : {scaler.__name__}yeo-johnson')
print(f'result : {model.score(x_test,y_test)}')

# scaler=PowerTransformer
# x_train,x_test=scaling_with_method(x_train,x_test,scaler,method='box-cox')
# model.fit(x_train,y_train)
# print(f'scaler : {scaler.__name__}box-cox')
# print(f'result : {model.score(x_test,y_test)}')
