import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris,load_digits,load_breast_cancer,fetch_covtype,load_wine
import pandas as pd
# x,y=load_digits(return_X_y=True)

# print(x.shape,np.unique(y))
# lda=LDA(n_components=9)
# x_lda=lda.fit_transform(x,y)
# lda_EVR =lda.explained_variance_ratio_

# cumsum=np.cumsum(lda_EVR)
# print(cumsum)


def EVR(x,y,name:str()):
    lda=LDA(n_components=min(x.shape[1]-1,len(np.unique(y))-1))
    x_lda=lda.fit_transform(x,y)
    lda_EVR =lda.explained_variance_ratio_

    cumsum=np.cumsum(lda_EVR)
    print('##################################################')
    print(f'x.shape[1]:{x.shape[1]} len(np.unique(y)):{len(np.unique(y))}')
    print(f'{name} {x.shape}->{x_lda.shape}\n:{cumsum}')

for dataset in [load_iris,load_digits,load_wine,load_breast_cancer,fetch_covtype]:
    x,y=dataset(return_X_y=True)
    EVR(x,y,dataset.__name__)
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
x=df.drop([df.columns[-1]],axis=1)
y=np.array(df[[df.columns[-1]]]).reshape((-1,))
EVR(x,y,'dacon_diabete')
path='./_data/dacon_wine/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
x=df.drop([df.columns[-1]],axis=1)
y=np.array(df[[df.columns[-1]]]).reshape((-1,))
EVR(x,y,'dacon_wine')
