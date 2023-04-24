import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
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

################################################################################################
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
################################################################################################
def plot_feature_importances(datasets=load_iris(),model=DecisionTreeClassifier()):
    n_features=datasets.data.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_,align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1,n_features)
    plt.title(type(model).__name__)
################################################################################################
model1 = DecisionTreeClassifier
model2 = RandomForestClassifier
model3 = GradientBoostingClassifier
model4 = XGBClassifier
x_train,x_test,y_train,y_test=train_test_split(*load_iris(return_X_y=True),train_size=0.8)
for i,v in enumerate([model1,model2,model3,model4]):
    plt.subplot(2,2,i+1)
    model=runmodel(x_train,x_test,y_train,y_test,model=v)
    plot_feature_importances(model=model)
plt.show()

################################################################################################
# # 2. model build
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('classifier', RandomForestClassifier())
# ])

# # 3. compile,training
# pipe.fit(x_train,y_train)

# # 4. evaluate
# print(f'model score : {pipe.score(x_test,y_test)}\nacc : {accuracy_score(y_test,pipe.predict(x_test))}')
################################################################################################
