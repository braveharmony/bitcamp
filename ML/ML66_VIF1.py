import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import fetch_california_housing

data={'size':[30,35,40,45,50,45],
      'rooms':[2,2,3,3,4,3],
      'window':[2,2,3,3,4,3],
      'year':[2010,2015,2010,2015,2010,2014],
      'price':[1.5,1.8,2.0,2.2,2.5,2.3]}
df=pd.DataFrame(data)
print(df)

scaler=StandardScaler()
x=df.drop(df.columns[-1],axis=1)
y=df[df.columns[-1]]
x_scaled=pd.DataFrame(scaler.fit_transform(x),columns=x.columns,index=x.index)
print(x_scaled)

vif=pd.DataFrame()
vif['variables']=x_scaled.columns
vif['vif']=[variance_inflation_factor(x_scaled,i) for i,v in enumerate(x_scaled.columns)]
print(vif)

print('===============rooms 제거전================')
model=LinearRegression()
model.fit(x_scaled,y)
print(f'결정계수 : {r2_score(y,model.predict(x_scaled))}')


x_scaled=x_scaled.drop(['rooms'],axis=1)
vif=pd.DataFrame()
vif['variables']=x_scaled.columns
vif['vif']=[variance_inflation_factor(x_scaled,i) for i,v in enumerate(x_scaled.columns)]
print(vif)
print('===============rooms 제거후================')
model=LinearRegression()
model.fit(x_scaled,y)
print(f'결정계수 : {r2_score(y,model.predict(x_scaled))}')
