import numpy as np
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

#==============================================================================================
def runmodel(x_train,x_test,y_train,y_test,model=DecisionTreeRegressor):
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)

    # 2. model build
    model=model()

    # 3. compile, training
    model.fit(x_train,y_train)

    # 4. predict,evaluate
    print('============================================================')
    print(f'{type(model).__name__} : {np.round(model.feature_importances_,2)}')
    print(f'model score : {model.score(x_test,y_test)}')
#==============================================================================================

for i in [fetch_california_housing,load_diabetes]:
    # 1. data prepare
    x,y=i(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed)
    for j in (DecisionTreeRegressor,GradientBoostingRegressor,RandomForestRegressor,XGBRegressor):
        # 2,3,4 model build, compile, training, predict, evaluate
        runmodel(x_train,x_test,y_train,y_test,model=j)
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
