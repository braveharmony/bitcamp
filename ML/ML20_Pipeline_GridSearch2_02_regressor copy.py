import numpy as np
from sklearn.datasets import fetch_california_housing,load_diabetes
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split,GridSearchCV,KFold
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
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
    pipe=Pipeline((('scaler',scaler()),('pca',PCA(x_train.shape[1]//2)),('rf',regressor())))
    def mergedict(a,b):
        parameterlists = [{"rf__n_estimators":[100,200]}]
        parameterlists.append({'rf__max_depth':[6,8,10,12]})
        parameterlists.append({'rf__min_samples_leaf':[3,5,7,10]})
        parameterlists.append({'rf__min_samples_split':[2,3,5,10]})
        parameterlists.append({'rf__n_jobs':[-1,2,4]})
        return {**parameterlists[a],**parameterlists[b]}
    parameters=[]
    parameters.append(mergedict(0,2))
    parameters.append(mergedict(1,2))
    parameters.append(mergedict(2,3))
    parameters.append(mergedict(3,4))
    parameters.append(mergedict(1,4))

    model=GridSearchCV(pipe,parameters,cv=KFold(5,shuffle=True,random_state=seed),verbose=1,n_jobs=-1)
    # 3. compile,training
    model.fit(x_train,y_train)
    # 4. evaluate,predict
    print('================================================================================')
    print(f'scaler : {scaler.__name__} regressor : {regressor.__name__}\nmodel score : {model.score(x_test,y_test)}\nr2_score : {r2_score(y_test,model.predict(x_test))}')
    return model.score(x_test,y_test),model.best_params_
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
    max_para=0
    x,y=i(return_X_y=True)
    for j in [MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler] :
        for k in [RandomForestRegressor]:
            current_score,current_para=pipeline(x,y,scaler=j,regressor=k)
            if max_score<current_score:
                max_score=current_score
                max_scaler=j
                max_model=k
                max_para=current_para
    results+='===============================================================\n'
    results+=i.__name__+' '
    results+=str(result(max_model,max_scaler,max_score))
    results+=str(max_para)+'\n'
print(results)
#==============================================================================================