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
import matplotlib as mpl
import matplotlib.pyplot as plt
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

#==============================================================================================
def runmodel(x_train,x_test,y_train,y_test,model=DecisionTreeClassifier):
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

def plot_feature_importances(datasets=load_iris(),model=DecisionTreeClassifier()):
    n_features=datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(type(model).__name__)
k=1
for i in [load_breast_cancer]:
    # 1. data prepare
    x,y=i(return_X_y=True)
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
    for j in (DecisionTreeClassifier,GradientBoostingClassifier,RandomForestClassifier,XGBClassifier):
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

# ============================================================
# DecisionTreeClassifier : [0.02 0.   0.03 0.96]
# model score : 0.9666666666666667
# data : load_iris
# ============================================================
# GradientBoostingClassifier : [0.01 0.01 0.16 0.82]
# model score : 0.9666666666666667
# data : load_iris
# ============================================================
# RandomForestClassifier : [0.09 0.02 0.42 0.47]
# model score : 0.9333333333333333
# data : load_iris
# ============================================================
# XGBClassifier : [0.02 0.02 0.59 0.37]
# model score : 0.9333333333333333
# data : load_iris
# ============================================================
# DecisionTreeClassifier : [0.   0.01 0.   0.   0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.04
#  0.01 0.   0.   0.   0.01 0.01 0.12 0.06 0.   0.   0.   0.   0.   0.73
#  0.01 0.  ]
# model score : 0.9210526315789473
# data : load_breast_cancer
# ============================================================
# GradientBoostingClassifier : [0.   0.   0.   0.   0.   0.   0.   0.11 0.   0.   0.   0.   0.   0.02
#  0.   0.   0.   0.   0.   0.   0.32 0.06 0.19 0.01 0.01 0.   0.01 0.24
#  0.01 0.  ]
# model score : 0.9473684210526315
# data : load_breast_cancer
# ============================================================
# RandomForestClassifier : [0.05 0.01 0.06 0.02 0.01 0.01 0.08 0.12 0.   0.   0.03 0.01 0.01 0.03
#  0.   0.   0.   0.   0.   0.   0.09 0.01 0.12 0.09 0.01 0.02 0.05 0.13
#  0.01 0.01]
# model score : 0.9385964912280702
# data : load_breast_cancer
# ============================================================
# XGBClassifier : [0.   0.01 0.   0.01 0.   0.   0.   0.05 0.   0.   0.01 0.01 0.01 0.01
#  0.   0.02 0.   0.   0.   0.01 0.18 0.03 0.33 0.01 0.   0.   0.02 0.27
#  0.   0.01]
# model score : 0.9649122807017544
# data : load_breast_cancer
# ============================================================
# DecisionTreeClassifier : [0.05 0.   0.02 0.02 0.   0.   0.13 0.   0.   0.34 0.   0.04 0.4 ]
# model score : 0.8888888888888888
# data : load_wine
# ============================================================
# GradientBoostingClassifier : [0.01 0.01 0.03 0.01 0.01 0.   0.26 0.   0.   0.34 0.   0.01 0.3 ]
# model score : 0.9722222222222222
# data : load_wine
# ============================================================
# RandomForestClassifier : [0.12 0.04 0.01 0.02 0.03 0.06 0.15 0.01 0.03 0.16 0.07 0.1  0.2 ]
# model score : 1.0
# data : load_wine
# ============================================================
# XGBClassifier : [0.02 0.05 0.05 0.01 0.03 0.02 0.27 0.   0.   0.26 0.05 0.01 0.23]
# model score : 0.9722222222222222
# data : load_wine
# ============================================================
# DecisionTreeClassifier : [0.   0.   0.02 0.01 0.01 0.06 0.   0.   0.   0.   0.02 0.   0.01 0.01
#  0.   0.   0.   0.   0.01 0.03 0.04 0.09 0.   0.   0.   0.   0.06 0.04
#  0.05 0.02 0.   0.   0.   0.06 0.01 0.01 0.08 0.02 0.01 0.   0.   0.01
#  0.08 0.07 0.   0.   0.   0.   0.   0.   0.01 0.   0.   0.01 0.03 0.
#  0.   0.   0.01 0.   0.06 0.03 0.   0.  ]
# model score : 0.8666666666666667
# data : load_digits
# ============================================================
# GradientBoostingClassifier : [0.   0.   0.01 0.   0.   0.06 0.01 0.   0.   0.01 0.02 0.   0.01 0.02
#  0.   0.   0.   0.   0.01 0.04 0.02 0.08 0.01 0.   0.   0.   0.06 0.01
#  0.04 0.02 0.01 0.   0.   0.06 0.01 0.01 0.07 0.01 0.02 0.   0.   0.01
#  0.08 0.07 0.01 0.02 0.02 0.   0.   0.   0.   0.02 0.01 0.01 0.03 0.
#  0.   0.   0.02 0.   0.06 0.01 0.03 0.  ]
# model score : 0.9555555555555556
# data : load_digits
# ============================================================
# RandomForestClassifier : [0.   0.   0.02 0.01 0.01 0.02 0.01 0.   0.   0.01 0.02 0.01 0.01 0.03
#  0.01 0.   0.   0.01 0.02 0.03 0.03 0.05 0.01 0.   0.   0.01 0.04 0.02
#  0.03 0.02 0.03 0.   0.   0.03 0.03 0.02 0.04 0.02 0.03 0.   0.   0.01
#  0.04 0.04 0.02 0.02 0.02 0.   0.   0.   0.02 0.02 0.01 0.02 0.03 0.
#  0.   0.   0.02 0.01 0.02 0.03 0.02 0.  ]
# model score : 0.975
# data : load_digits
# ============================================================
# XGBClassifier : [0.   0.04 0.01 0.01 0.   0.04 0.01 0.03 0.   0.01 0.01 0.01 0.01 0.01
#  0.   0.   0.   0.01 0.01 0.05 0.01 0.04 0.01 0.   0.   0.01 0.03 0.01
#  0.03 0.02 0.01 0.   0.   0.07 0.01 0.   0.06 0.01 0.04 0.   0.   0.01
#  0.03 0.04 0.01 0.02 0.03 0.   0.   0.   0.01 0.02 0.01 0.01 0.03 0.
#  0.   0.   0.01 0.01 0.07 0.02 0.04 0.02]
# model score : 0.9611111111111111
# data : load_digits