from sklearn.datasets import load_breast_cancer,load_diabetes,fetch_california_housing,load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from typing import Tuple,Dict
import os
from sklearn.metrics import r2_score,mean_squared_error
import numpy as np
def runXGBClassifier(x,y)->Tuple[XGBClassifier,dict]:
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=20580,train_size=0.8)

    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    n_splits=5
    kfold=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

    parameters={'n_estimators' : 1000,#디폴트 100,1~inf,int
                'learning_rate' : 0.3,#디폴트 0.3/0~1/float
                'max_depth':10,#디폴트 6/0~inf/int
                'gamma':1,#디폴트 0/0~inf/float
                'min_child_weight':1,#디폴트 1/0~inf/float
                'subsample':1,#디폴트 1/0~1/float
                'colsample_bytree':1,#디폴트 1/0~1/float
                'colsample_bylevel':1,#디폴트 1/0~1/float
                'colsample_bynode':1,#디폴트 1/0~1/float
                'reg_alpha':0,#디폴트 0/0~inf/float
                'reg_lambda':1,#디폴트 1/0~inf/float
                'random_state':0
                }
    
    
    model=XGBRegressor(
        # **parameters
                       )
    model.fit(x_train,y_train
            #   ,early_stopping_rounds=500
              ,eval_set=[(x_test,y_test)],verbose=0
            #   ,eval_metric='logloss'#이진분류
            #   ,eval_metric='error'#이진분류
            #   ,eval_metric='auc'#이진분류
            #   ,eval_metric='merror'#다중분류 mlogloss
            #   ,eval_metric='rmse'#,'mae','rmsle',...#회귀
              )
    # print(f'최상의 매개변수 : {model.best_params_}')
    # print(f'최상의 점수 : {model.best_score_}')
    # print(f'test score : {model.score(x_test,y_test)}')
    # hist=model.evals_result()
    hist=None
    return model,hist
def runwithdrop(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=20580,train_size=0.8)

    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    model,hist=runXGBClassifier(x,y)


    from sklearn.feature_selection import SelectFromModel
    threshold=np.sort(model.feature_importances_)
    for i in threshold:
        selection=SelectFromModel(model,threshold=i,prefit=True)
        select_x=selection.transform(x)
        x_train,x_test,y_train,y_test = train_test_split(select_x,y,random_state=20580,train_size=0.8)

        scaler=RobustScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        selected_model,hist=runXGBClassifier(select_x,y)     
        print(f'변형된 x_train : {x_train.shape} 변형된 x_test : {x_test.shape}')
        y_predict=selected_model.predict(x_test)
        r2=r2_score(y_test,y_predict)
        rmse=np.sqrt(mean_squared_error(y_test,y_predict))
        print(f'결정계수: {r2}\nrmse : {rmse}')
    return x,y,model
x,y=load_diabetes(return_X_y=True)
x,y,model=runwithdrop(x,y)

# for i in range(x.shape[1]):
#     print(f'{i}번 삭제')
#     x,y,model=runwithdrop(x,y)