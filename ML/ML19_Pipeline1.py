import numpy as np
from sklearn.datasets import load_iris,load_digits
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
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
    ('classifier', SVC())
])

# 3. compile,training
pipe.fit(x_train,y_train)

# 4. evaluate
print(f'model score : {pipe.score(x_test,y_test)}\nacc : {accuracy_score(y_test,pipe.predict(x_test))}')
#==============================================================================================

# #==============================================================================================
# # 2. model build
# model=make_pipeline(MinMaxScaler(),RandomForestClassifier())
# # 3. compile,training
# model.fit(x_train,y_train)
# # 4. evaluate,predict
# print(f'model score : {model.score(x_test,y_test)}\nacc : {accuracy_score(y_test,model.predict(x_test))}')
# #==============================================================================================
