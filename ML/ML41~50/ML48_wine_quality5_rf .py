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
df=df.dropna()
df=df[~df.quality.isin([3, 4, 8, 9])]
x=df.drop([df.columns[0]], axis=1)
y=df[df.columns[0]]
y-=np.min(y)
print(np.unique(y,return_counts=True))
unique_classes, counts = np.unique(y, return_counts=True)

print(y.shape)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)

# scaler=RobustScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
# dft=scaler.transform(dft)

print(y_train.shape)
# 2. model
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50000)
model.fit(x_train,y_train)

y_pred=model.predict(dft)+3
dfs[dfs.columns[-1]]=y_pred
print(model.score(x_test,y_test))
# print(y_pred)
import datetime
now=datetime.datetime.now().strftime('%m월%d일 %H시%M분')
# dfs.to_csv(f'./_save/dacon_wine/{now}_sub.csv',index=False)