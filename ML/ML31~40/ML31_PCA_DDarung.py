import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import random
# 0. seed initialization
seed=4
random.seed(seed)
np.random.seed(seed)

def run_model(x,y,label:str=''):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(n_estimators=200,max_depth=20)
    model.fit(x_train,y_train)
    if label!='':
        print(f'{label} 결과')
    print(f'model score : {model.score(x_test,y_test)}')
    

path='./_data/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
x=df.drop([df.columns[-1]],axis=1)
y=np.array(df[[df.columns[-1]]]).reshape((-1,))

run_model(x,y,'PCA이전')

pca = PCA(n_components=7)
print(x.shape)
x=pca.fit_transform(x)
print(x.shape)

run_model(x,y,'PCA이후')
