import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.datasets import load_iris,load_breast_cancer,load_digits,load_wine
from sklearn.datasets import fetch_california_housing,load_diabetes
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.preprocessing import RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler

# 1. data prepare
def run_model_with_drop(dataset_class=load_iris
                        ,model_class=RandomForestRegressor
                        ,scaler_class=StandardScaler
                        ):
    print('======================================================')
    print(f'dataset name : {dataset_class.__name__}')
    print(f'model name : {model_class.__name__}')
    print(f'scaler name : {scaler_class.__name__}')
    datasets=dataset_class()
    df=pd.DataFrame(datasets.data,columns=datasets.feature_names)
    df['target']=datasets.target
    print(df)


    y=df['target']
    x=df.drop(['target'],axis=1)
    scaler=scaler_class()
    x_scaled=scaler.fit_transform(x)

    vif=pd.DataFrame()
    vif['variables']=x.columns
    vif['VIF']=[variance_inflation_factor(x_scaled,i)for i in range(len(x.columns))]
    print(vif)


    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
    scaler=scaler_class()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    model=model_class()
    model.fit(x_train,y_train)
    print(f'score : {r2_score(y_test,model.predict(x_test))}')
    
    
    
    col_drop=vif['variables'][np.argmax(vif['VIF'])]
    print(f'드랍할 컬럼:{col_drop}')
    
    x=x.drop(col_drop,axis=1)
    vif=pd.DataFrame()
    vif['variables']=x.columns
    vif['VIF']=[variance_inflation_factor(x_scaled,i)for i in range(len(x.columns))]
    print(vif)
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
    scaler=scaler_class()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    model=model_class()
    model.fit(x_train,y_train)
    print(f'score : {r2_score(y_test,model.predict(x_test))}')



CF_dataset_classes=[load_iris,load_breast_cancer,load_digits,load_wine]
RE_dataset_classes=[fetch_california_housing,load_diabetes]
CF_model_classes=[LogisticRegression,DecisionTreeClassifier,RandomForestClassifier]
RE_model_classes=[LinearRegression,DecisionTreeRegressor,RandomForestRegressor]
scaler_classes=[RobustScaler,MinMaxScaler,StandardScaler,MaxAbsScaler]
for dataset_class in CF_dataset_classes:
    for model_class in CF_model_classes:
        for scaler_class in scaler_classes:
            run_model_with_drop(dataset_class,model_class,scaler_class)
for dataset_class in RE_dataset_classes:
    for model_class in RE_model_classes:
        for scaler_class in scaler_classes:
            run_model_with_drop(dataset_class,model_class,scaler_class)