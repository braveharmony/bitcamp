# 1. data prepare
import numpy as np
from sklearn.datasets import load_iris,load_digits
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
import random
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

# 1. data prepare
x,y=load_digits(return_X_y=True)

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

# #==============================================================================================
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)

# pca=PCA(n_components=8)
# x_train=pca.fit_transform(x_train)
# x_test=pca.transform(x_test)

# # 2. model build
# model=SVC()

# # 3. compile, training
# model.fit(x_train,y_train)

# # 4. predict,evaluate
# print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
# #==============================================================================================


#==============================================================================================
# 2. model build
pipe = Pipeline([('scaler', StandardScaler()),
    ('pca',PCA(n_components=8)),
    ('rf', RandomForestClassifier())
])
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

model=GridSearchCV(pipe,parameters,cv=StratifiedKFold(5,shuffle=True,random_state=seed),verbose=1,n_jobs=-1)


# 3. compile,training
model.fit(x_train,y_train)

# 4. evaluate
print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
#==============================================================================================

# #==============================================================================================
# # 2. model build
# model=make_pipeline(MinMaxScaler(),RandomForestClassifier())
# # 3. compile,training
# model.fit(x_train,y_train)
# # 4. evaluate,predict
# print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
# #==============================================================================================
