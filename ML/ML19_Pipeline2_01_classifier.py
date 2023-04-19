import numpy as np
from sklearn.datasets import load_iris,load_breast_cancer,load_digits
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from sklearn.model_selection import train_test_split
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
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
# model=RandomForestClassifier()

# # 3. compile, training
# model.fit(x_train,y_train)

# # 4. predict,evaluate
# print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
# #==============================================================================================


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

#==============================================================================================
def pipeline(x,y,scaler=MinMaxScaler,classifier=RandomForestClassifier):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)    
    # 2. model build
    model=Pipeline((('scaler',scaler()),('pca',PCA(x_train.shape[1]//2)),('classifier',classifier())))
    # 3. compile,training
    model.fit(x_train,y_train)
    # 4. evaluate,predict
    print('================================================================================')
    print(f'scare : {scaler.__name__} classifier : {classifier.__name__}\nmodel score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
    return model.score(x_test,y_test)
#==============================================================================================
class result:
    def __init__(self,model,scaler,score):
        self.model=model
        self.scaler=scaler
        self.score=score
    def __str__(self):
        return f'model : {self.model.__name__} scaler : {self.scaler.__name__} score : {self.score}\n'
#==============================================================================================
# 1. data prepare
results=str()
for i in [load_iris,load_digits,load_breast_cancer]:
    max_score=0
    max_scaler=str()
    max_model=str()
    x,y=i(return_X_y=True)
    for j in [MinMaxScaler,MaxAbsScaler,StandardScaler,RobustScaler]:
        for k in [RandomForestClassifier,SVC,DecisionTreeClassifier]:
            current_score=pipeline(x,y,scaler=j,classifier=k)
            if max_score<current_score:
                max_score=current_score
                max_scaler=j
                max_model=k
    results+='===============================================================\n'
    results+=i.__name__+' '
    results+=str(result(max_model,max_scaler,max_score))
print(results)
#==============================================================================================
