# 0. seed
import random
import numpy as np
import tensorflow as tf
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. 데이터
import pandas as pd
path="./_data/dacon_wine/"
path_save='./_save/dacon_wine/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')

from sklearn.preprocessing import MinMaxScaler,LabelEncoder,RobustScaler
le=LabelEncoder()
df[df.columns[-1]]=le.fit_transform(df[df.columns[-1]])
dft[df.columns[-1]]=le.transform(dft[df.columns[-1]])

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from xgboost import XGBRegressor
imputer=IterativeImputer(XGBRegressor())
df=pd.DataFrame(imputer.fit_transform(df),columns=df.columns,index=df.index)
df=df.interpolate().dropna()
x=df.drop([df.columns[0]], axis=1)
y=df[df.columns[0]]
print(df.groupby(df.columns[0]).mean())
df_group=pd.DataFrame(df.groupby(df.columns[0]).mean())
import matplotlib.pyplot as plt
for i,v in enumerate(df_group.columns):
    plt.subplot(3,4,i+1)
    plt.bar(x=df_group.index,height=df_group[v],label=v)
    plt.legend()
    plt.grid()
plt.show()