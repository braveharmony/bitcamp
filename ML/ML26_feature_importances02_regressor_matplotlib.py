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
import matplotlib as mpl
import matplotlib.pyplot as plt
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
    return model
#==============================================================================================
def plot_feature_importances(datasets=fetch_california_housing(),model=DecisionTreeRegressor()):
    n_features=datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(type(model).__name__)
k=1
for i in [fetch_california_housing]:
    # 1. data prepare
    x,y=i(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed)
    for j in (DecisionTreeRegressor,GradientBoostingRegressor,RandomForestRegressor,XGBRegressor):
        # 2,3,4 model build, compile, training, predict, evaluate
        model=runmodel(x_train,x_test,y_train,y_test,model=j)
        print(f'data : {i.__name__}')
        datasets=i()
        plt.subplot(2,2,k)
        k+=1
        plot_feature_importances(datasets,model)
plt.show()

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
