import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_wine,load_digits
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

# 1. data prepare
for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
    x,y=i(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)


#==============================================================================================
def runmodel(x_train,x_test,y_train,y_test,model=DecisionTreeClassifier):
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)


    # 2. model build
    # model=XGBClassifier()
    # model=RandomForestClassifier()
    model=model()

    # 3. compile, training
    model.fit(x_train,y_train)

    # 4. predict,evaluate
    print('============================================================')
    print(f'{type(model).__name__} : {np.round(model.feature_importances_,2)}')
    print(f'model score : {model.score(x_test,y_test)}')
#==============================================================================================
for i in [load_iris,load_breast_cancer,load_wine,load_digits]:
    x,y=i(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
    for j in (DecisionTreeClassifier,GradientBoostingClassifier,RandomForestClassifier,XGBClassifier):
        runmodel(x_train,x_test,y_train,y_test,j)
        print(f'data : {i.__name__}')


# #==============================================================================================
# # 2. model build
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', RandomForestClassifier())
# ])

# # 3. compile,training
# pipe.fit(x_train,y_train)

# # 4. evaluate
# print(f'model score : {pipe.score(x_test,y_test)}\nacc : {accuracy_score(y_test,pipe.predict(x_test))}')
# #==============================================================================================

# #==============================================================================================
# # 2. model build
# model=make_pipeline(MinMaxScaler(),RandomForestClassifier())
# # 3. compile,training
# model.fit(x_train,y_train)
# # 4. evaluate,predict
# print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
# #==============================================================================================
