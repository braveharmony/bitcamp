import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler,MinMaxScaler

# 1. data prepare
datasets=fetch_california_housing()
df=pd.DataFrame(datasets.data,columns=datasets.feature_names)
df['target']=datasets.target
print(df)
# df.boxplot()
def print_box(df:pd.DataFrame):
    scaler=MinMaxScaler()
    df=pd.DataFrame(scaler.fit_transform(df),columns=df.columns,index=df.index)
    df.boxplot()
    plt.show()
print(df.info())
print(df.describe())
# df['Population'].hist(bins=50)
# plt.show()

y=df['target']
x=df.drop(['target'],axis=1)
x['Population']=np.log1p(np.min(x['Population']))
# x['Population'].hist(bins=50)
# plt.show()
# y=np.log1p(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
y_train=np.log1p(y_train)


model=RandomForestRegressor(random_state=0)

model.fit(x_train,y_train)
from sklearn.metrics import r2_score
print(f'score : {r2_score(y_test,np.expm1(model.predict(x_test)))}')