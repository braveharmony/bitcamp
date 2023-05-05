import numpy as np
import pandas as pd
from sklearn.datasets import load_iris,load_diabetes,load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import f1_score
# 0. seed initialization
seed=4
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,label:str=''):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(n_estimators=200,max_depth=20)
    model.fit(x_train,y_train)
    if label!='':
        print(f'{label} 결과')
    print(f'f1 score : {f1_score(y_test,model.predict(x_test))}')
    print(f'model score : {model.score(x_test,y_test)}')
    return model.score(x_test,y_test)

dataset=load_breast_cancer()

x=dataset['data']
y=dataset['target']

run_model(x,y,'PCA이전')
max_score=0
max_index=0
for i in range(30):
    pca = PCA(n_components=30-i)
    x_PCA=pca.fit_transform(x)
    print(f'current index : {i}')
    current_score=run_model(x_PCA,y)
    if current_score>max_score:
        max_score=current_score
        max_index=i
    print('##########################################################################')
print(f'max index : {max_index}\nmax_score : {max_score}')
