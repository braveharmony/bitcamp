import numpy as np
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)


# #==============================================================================================
# scaler=MinMaxScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

# # 2. model build
# model=RandomForestRegressor()

# # 3. compile, training
# model.fit(x_train,y_train)

# # 4. predict,evaluate
# print(f'model score : {model.score(x_test,y_test)}\nacc : {r2_score(y_test,model.predict(x_test))}')
# #==============================================================================================


# #==============================================================================================
# # 2. model build
# pipe = Pipeline([
#     ('scaler', StandardScaler()),
#     ('regressor', RandomForestRegressor())
# ])

# # 3. compile,training
# pipe.fit(x_train,y_train)

# # 4. evaluate
# print(f'model score : {pipe.score(x_test,y_test)}\nacc : {r2_score(y_test,pipe.predict(x_test))}')
# #==============================================================================================

#==============================================================================================
def pipeline(x,y,scaler=MinMaxScaler,regressor=RandomForestRegressor):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed)    
    # 2. model build
    model=make_pipeline(scaler(),regressor())
    # 3. compile,training
    model.fit(x_train,y_train)
    # 4. evaluate,predict
    print('================================================================================')
    print(f'scaler : {scaler.__name__} regressor : {regressor.__name__}\nmodel score : {model.score(x_test,y_test)}\nr2_score : {r2_score(y_test,model.predict(x_test))}')
    return model.score(x_test,y_test)
#==============================================================================================

# 1. data prepare
class result:
    def __init__(self,model,scaler,score):
        self.model=model
        self.scaler=scaler
        self.score=score
    def __str__(self):
        return f'model : {self.model.__name__} scaler : {self.scaler.__name__} score : {self.score}\n'
    def __call__(self):
        return f'model : {self.model.__name__} scaler : {self.scaler.__name__} score : {self.score}\n'
    def string(self):
        return f'model : {self.model.__name__} scaler : {self.scaler.__name__} score : {self.score}\n'

#==============================================================================================
results=str()
for i in [fetch_california_housing,load_diabetes]:
    max_score=0
    max_scaler=str()
    max_model=str()
    x,y=i(return_X_y=True)
    for j in [MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler] :
        for k in [RandomForestRegressor,LinearSVR,DecisionTreeRegressor]:
            current_score=pipeline(x,y,scaler=j,regressor=k)
            if max_score<current_score:
                max_score=current_score
                max_scaler=j
                max_model=k
    results+='===============================================================\n'
    results+=str(result(max_model,max_scaler,max_score))
print(results)
#==============================================================================================