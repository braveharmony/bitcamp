from sklearn.datasets import load_breast_cancer,load_diabetes,fetch_california_housing,load_digits
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier,XGBRegressor
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
def runXGBRegressor(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8,stratify=y)

    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    n_splits=5
    kfold=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

    parameters={'n_estimators' : [100],#디폴트 100,1~inf,int
                'learning_rate' : [0.1,0.2,0.3,0.5,1,0.01,0.001],#디폴트 0.3/0~1/float
                'max_depth':[2,3,4,5,6,7,8,9,10],#디폴트 6/0~inf/int
                'gamma':[0,1,2,3,4,5,6,7,8,9,10],#디폴트 0/0~inf/float
                'min_child_weight':[0,0.01,0.1,0.2,0.5,1,2,3,5,10,100],#디폴트 1/0~inf/float
                'subsample':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bytree':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bylevel':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bynode':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'reg_alpha':[0,0.01,0.1,0.2,0.5,1,2,5,10],#디폴트 0/0~inf/float
                'reg_lambda':[0,0.01,0.1,0.2,0.5,1,2,5,10]#디폴트 1/0~inf/float
                }

    xgb=XGBRegressor(random_state=0)
    model=RandomizedSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
    model.fit(x_train,y_train)
    print(f'최상의 매개변수 : {model.best_params_}')
    print(f'최상의 점수 : {model.best_score_}')
    print(f'test score : {model.score(x_test,y_test)}')
    return model
def runXGBClassifier(x,y):
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8,stratify=y)

    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    n_splits=5
    kfold=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=0)

    parameters={'n_estimators' : [100],#디폴트 100,1~inf,int
                'learning_rate' : [0.1,0.2,0.3,0.5,1,0.01,0.001],#디폴트 0.3/0~1/float
                'max_depth':[2,3,4,5,6,7,8,9,10],#디폴트 6/0~inf/int
                'gamma':[0,1,2,3,4,5,6,7,8,9,10],#디폴트 0/0~inf/float
                'min_child_weight':[0,0.01,0.1,0.2,0.5,1,2,3,5,10,100],#디폴트 1/0~inf/float
                'subsample':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bytree':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bylevel':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'colsample_bynode':[0,0.01,0.1,0.2,0.5,1],#디폴트 1/0~1/float
                'reg_alpha':[0,0.01,0.1,0.2,0.5,1,2,5,10],#디폴트 0/0~inf/float
                'reg_lambda':[0,0.01,0.1,0.2,0.5,1,2,5,10]#디폴트 1/0~inf/float
                }

    xgb=XGBClassifier(random_state=0)
    model=RandomizedSearchCV(xgb,parameters,cv=kfold,n_jobs=-1)
    model.fit(x_train,y_train)
    print(f'최상의 매개변수 : {model.best_params_}')
    print(f'최상의 점수 : {model.best_score_}')
    print(f'test score : {model.score(x_test,y_test)}')
    return model
x,y=load_breast_cancer(return_X_y=True)
runXGBClassifier(x,y)
x,y=load_digits(return_X_y=True)
runXGBClassifier(x,y)
x,y=load_diabetes(return_X_y=True)
runXGBRegressor(x,y)
x,y=fetch_california_housing(return_X_y=True)
runXGBRegressor(x,y)
