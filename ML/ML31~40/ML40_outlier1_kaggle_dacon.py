from ML39_outlier2_반장 import find_2d_outliers,index_2d_outliers,remove_2d_outliers
import pandas as pd
import numpy as np

def runRegressor(df:pd.DataFrame):
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
        try:
            if type(i)==pd.DataFrame:
                runpandasmodel(i)
            else:
                runnumpymodel(i)
        except Exception as e:print(e)
            
def runClassifier(df:pd.DataFrame):
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
    from xgboost import XGBClassifier,XGBRegressor
    imputer=SimpleImputer(strategy='median')
    dfs.append(imputer.fit_transform(df))
    imputer=KNNImputer()
    dfs.append(imputer.fit_transform(df))
    imputer=IterativeImputer(estimator=XGBRegressor())
    dfs.append(imputer.fit_transform(df))

    def runpandasmodel(df:pd.DataFrame):
        model=XGBClassifier(tree_method='gpu_hist',
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
        model=XGBClassifier(tree_method='gpu_hist',
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
        try:
            if type(i)==pd.DataFrame:
                runpandasmodel(i)
            else:
                runnumpymodel(i)
        except Exception as e:print(e)
path='./_data/kaggle_bike/'
df=pd.read_csv(path+'train.csv',index_col=0)
x=df.drop(df.columns[-2:],axis=1)
y=df[df.columns[-1]].values.reshape(-1,1)
df=pd.DataFrame(np.concatenate((remove_2d_outliers(x),y),axis=1),columns=list(x.columns)+[df.columns[-1]])
# print(df)
# print(df.isna().sum())
runRegressor(df)

path='./_data/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
x=df.drop(df.columns[-1:],axis=1)
y=df[df.columns[-1]].values.reshape(-1,1)
df=pd.DataFrame(np.concatenate((remove_2d_outliers(x),y),axis=1),columns=df.columns)
# print(df)
# print(df.isna().sum())
runRegressor(df)

path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
x=df.drop(df.columns[-1:],axis=1)
y=df[df.columns[-1]].values.reshape(-1,1)
y-=np.min(y)
df=pd.DataFrame(np.concatenate((remove_2d_outliers(x),y),axis=1),columns=df.columns)
# print(df)
# print(df.isna().sum())
runClassifier(df)

path='./_data/dacon_wine/'
df=pd.read_csv(path+'train.csv',index_col=0)
x=df.drop([df.columns[0],df.columns[-1]],axis=1)
y=df[df.columns[0]].values.reshape(-1,1)
y-=np.min(y)
df=pd.DataFrame(np.concatenate((remove_2d_outliers(x),y),axis=1),columns=list(x.columns)+[df.columns[0]])
# print(df)
# print(df.isna().sum())
runClassifier(df)
