import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler

# 1. data prepare
datasets=fetch_california_housing()
df=pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target']=datasets.target
print(df)

y=df['target']
x=df.drop(['target'],axis=1)
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

vif=pd.DataFrame()
vif['variables']=x.columns
vif['VIF']=[variance_inflation_factor(x_scaled,i)for i in range(len(x.columns))]
print(vif)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=RandomForestRegressor(random_state=0)
model.fit(x_train,y_train)
from sklearn.metrics import r2_score
print(f'score : {r2_score(y_test,model.predict(x_test))}')


x=x.drop('Latitude',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=RandomForestRegressor(random_state=0)
model.fit(x_train,y_train)
from sklearn.metrics import r2_score
print(f'score : {r2_score(y_test,model.predict(x_test))}')


x=x.drop('Longitude',axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model=RandomForestRegressor(random_state=0)
model.fit(x_train,y_train)
from sklearn.metrics import r2_score
print(f'score : {r2_score(y_test,model.predict(x_test))}')