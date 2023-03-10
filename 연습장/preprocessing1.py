import tensorflow as tf
import numpy as np
import random, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/kaggle_bike/'
df=pd.read_csv(path+'train.csv')
dft=pd.read_csv(path+'test.csv')
dfs=pd.read_csv(path+'sampleSubmission.csv')
print(df.info())
print(dft.info())
print(dfs.info())
print(df[df.columns[0]])

####################데이터 프레임 1번째 추출해서 스페이스바 기준으로 나눔##########################
print(df[df.columns[0]])
df2=pd.DataFrame(df[df.columns[0]].str.split(" ").tolist(),columns=["date","time"])
####################데이터 프레임 1번째 추출해서 스페이스바 기준으로 나눔##########################

####################데이터 프레임2 1번째 추출해서 미들바 기준으로 나눔############################
df2ymd=pd.DataFrame(df2[df2.columns[0]].str.split("-").tolist(),columns=['year','month','date'])
####################데이터 프레임2 1번째 추출해서 미들바 기준으로 나눔############################

####################데이터 프레임2 2번째 추출해서 미들바 기준으로 나눔############################
df2hms=pd.DataFrame(df2[df2.columns[1]].str.split(':').tolist(),columns=['hour','min','sec'])
####################데이터 프레임2 2번째 추출해서 미들바 기준으로 나눔############################

##########################################합침#################################################
ndf=pd.concat([df2ymd,df2hms,df.drop([df.columns[0]],axis=1)],axis=1)
ndf=ndf.apply(pd.to_numeric)
##########################################합침#################################################

print(ndf)
print(ndf.info())
x=ndf.drop([ndf.columns[-1],'min','sec'],axis=1)
y=ndf[ndf.columns[-1]]
for i in x.columns:
    plt.figure(i)
    plt.scatter(x[i],y)
    plt.title(i)
plt.show()