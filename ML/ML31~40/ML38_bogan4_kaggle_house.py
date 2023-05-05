import pandas as pd
from xgboost import XGBRegressor
import numpy as np
# 1. data prepare
path = './_data/kaggle_house/'
df=pd.read_csv(path+'train.csv',index_col=0)

for i in df.columns:
    if df[i].dtypes=='object':
        df=df.drop([i],axis=1)

dfs=[]
dfs.append(df.dropna())
dfs.append(df.dropna(axis=0))
dfs.append(df.dropna(axis=1))
dfs.append(df.fillna(df.mean()))
dfs.append(df.fillna(df.median()))
dfs.append(df.fillna(method='ffill'))
dfs.append(df.fillna(method='bfill'))
dfs.append(df.interpolate().fillna(method='bfill'))

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from xgboost import XGBRegressor
imputer=SimpleImputer(strategy='median')
dfs.append(imputer.fit_transform(df))
imputer=KNNImputer()
dfs.append(imputer.fit_transform(df))
imputer=IterativeImputer(estimator=XGBRegressor())
dfs.append(imputer.fit_transform(df))

def runpandasmodel(df:pd.DataFrame):
    model=XGBRegressor(tree_method='gpu_hist',
                       predictor='gpu_predictor',
                       gpu_id=0,n_estimators=100,learning_rate=0.3,
                       max_depth=4)
    x=df.drop(df.columns[-1],axis=1)
    y=df[df.columns[-1]]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
    model.fit(x_train,y_train)
    print(model.score(x_test,y_test))
def runnumpymodel(df:np.ndarray):
    model=XGBRegressor(tree_method='gpu_hist',
                       predictor='gpu_predictor',
                       gpu_id=0,n_estimators=100,learning_rate=0.3,
                       max_depth=4)
    x=df[:,:-1]
    y=df[:,-1]
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
    model.fit(x_train,y_train)
    print(model.score(x_test,y_test))
for i in dfs:
    if type(i)==pd.DataFrame:
        runpandasmodel(i)
    else:
        runnumpymodel(i)