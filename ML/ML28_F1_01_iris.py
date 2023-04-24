# [실습]
# 피쳐 임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거
# 재구성 후
# 모델을 거쳐서 결과 도출

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

################################################################################################

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
    current_model=model()

    # 3. compile, training
    current_model.fit(x_train,y_train)
    print('============================================================')
    print(f'{type(current_model).__name__} : {np.round(current_model.feature_importances_,2)}')
    print(f'기존 score : {current_model.score(x_test,y_test)}')

    # 4. Get feature importances
    feature_importances = current_model.feature_importances_

    # 5. Get the indices of the columns to keep (top 75%)
    threshold = np.percentile(feature_importances, 25)
    indices_to_keep = np.where(feature_importances >= threshold)[0]

    # 6. Filter the training and testing data to keep only the top 75% features
    x_train_filtered = x_train[:, indices_to_keep]
    x_test_filtered = x_test[:, indices_to_keep]

    # 7. Initialize a new model instance for retraining with filtered data
    current_model = model()

    # 8. Retrain the model with filtered data
    current_model.fit(x_train_filtered, y_train)

    # 9. predict,evaluate
    print(f'{type(current_model).__name__} : {np.round(current_model.feature_importances_,2)}')
    print(f'최종 score : {current_model.score(x_test_filtered,y_test)}')
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
