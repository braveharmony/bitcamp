# Linear Discriminant Analysis
from sklearn.datasets import load_iris,load_digits,load_breast_cancer,fetch_covtype,load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.datasets import cifar10
import pandas as pd
import matplotlib.pyplot as plt

def plotLDA(x,y,name:str()):
    lda=LDA(n_components=min(2,x.shape[1]-1,len(np.unique(y))-1))
    lda.fit(x,y)
    x=lda.transform(x)
    model=XGBClassifier(tree_method='gpu_hist',
                        predictor='gpu_predictor',
                        gpu_id=0,n_estimators=100,learning_rate=0.3,
                        max_depth=4)
    model.fit(x,y)
    print(model.score(x,y))

    pred=model.predict(x)


    ######################챗 지 피 티#######################
    if x.shape[1]==2:
        # 각 y값에 대해 lda된 x의 시각화
        plt.figure(name)
        plt.subplot(1,2,1)
        for i in np.unique(y):
            plt.scatter(x[y==i, 0], x[y==i, 1], label=i)

        plt.legend(loc='best')
        plt.title(f"LDA Transformed Features for {name} Dataset")
        plt.xlabel("LDA Component 1")
        plt.ylabel("LDA Component 2")

        # 각 y값에 대해 XGBClassifier로 분류된 x의 시각화
        plt.subplot(1,2,2)
        for i in np.unique(y):
            plt.scatter(x[pred==i, 0], x[pred==i, 1], label=i)

        plt.legend(loc='best')
        plt.title(f"XGBClassifier Predictions for {name} Dataset")
        plt.xlabel("LDA Component 1")
        plt.ylabel("LDA Component 2")
    else: print(f'{name}의 LDA결과 : 1차원')
#####################################################
for dataset in [load_iris,load_digits,load_wine,load_breast_cancer]:
    x,y=dataset(return_X_y=True)
    plotLDA(x,y,dataset.__name__)
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
x=df.drop([df.columns[-1]],axis=1)
y=np.array(df[[df.columns[-1]]]).reshape((-1,))
plotLDA(x,y,'dacon_diabete')
# path='./_data/dacon_wine/'
# df=pd.read_csv(path+'train.csv',index_col=0)
# df=df.dropna()
# x=df.drop([df.columns[-1],df.columns[1]],axis=1)
# y=np.array(df[[df.columns[1]]]).reshape((-1,))
# plotLDA(x,y,'dacon_wine')
plt.show()