from sklearn.datasets import load_breast_cancer,load_diabetes,fetch_california_housing,load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
from typing import Tuple,Dict
import os
def runXGBClassifier(x,y)->Tuple[XGBClassifier,dict]:
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8,stratify=y)

    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    n_splits=5
    kfold=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

    parameters={'n_estimators' : 100,#디폴트 100,1~inf,int
                'learning_rate' : 1,#디폴트 0.3/0~1/float
                'max_depth':6,#디폴트 6/0~inf/int
                'gamma':1,#디폴트 0/0~inf/float
                'min_child_weight':1,#디폴트 1/0~inf/float
                'subsample':1,#디폴트 1/0~1/float
                'colsample_bytree':1,#디폴트 1/0~1/float
                'colsample_bylevel':1,#디폴트 1/0~1/float
                'colsample_bynode':0,#디폴트 1/0~1/float
                'reg_alpha':0,#디폴트 0/0~inf/float
                'reg_lambda':1,#디폴트 1/0~inf/float
                'random_state':337
                
                }
    
    
    model=XGBClassifier(**parameters)
    model.fit(x_train,y_train
            #   ,early_stopping_rounds=500
              ,eval_set=[(x_train,y_train),(x_test,y_test)],verbose=0
            #   ,eval_metric='logloss'#이진분류
            #   ,eval_metric='error'#이진분류
            #   ,eval_metric='auc'#이진분류
            #   ,eval_metric='merror'#다중분류 mlogloss
              ,eval_metric='rmse'#,'mae','rmsle',...#회귀
              )
    # print(f'최상의 매개변수 : {model.best_params_}')
    # print(f'최상의 점수 : {model.best_score_}')
    # print(f'test score : {model.score(x_test,y_test)}')
    hist=model.evals_result()
    return model,hist
x,y=load_breast_cancer(return_X_y=True)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8,stratify=y)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model,hist=runXGBClassifier(x,y)

path='./_save/save_model_test/'
os.makedirs(path,exist_ok=True)
model.save_model(path+'ML43_save_model1_save.dat')
print(model.score(x_test,y_test))