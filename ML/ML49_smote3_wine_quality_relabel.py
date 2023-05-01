import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE

#1.Data prepare
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
x=df.drop([df.columns[0]], axis=1)
y=df[df.columns[0]]
y-=np.min(y)
print(np.unique(y,return_counts=True))
for i in range(len(y)):
    if y[i] in [0,1,2]:
        y[i]=0
    elif y[i]==3:
        y[i]=1
    else: y[i]=2

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,shuffle=True,random_state=3377,stratify=y)
print(x_train.shape,y_train.shape)

smote=SMOTE()
x_train,y_train=smote.fit_resample(x_train,y_train)
print(x_train.shape,y_train.shape)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=3377)

model.fit(x_train,y_train)
print(f'model score : {model.score(x_test,y_test)}')
print(f'acc score : {accuracy_score(y_test,model.predict(x_test))}')
print(f"f1_score(macro) score : {f1_score(y_test,model.predict(x_test),average='macro')}")
print(f"f1_score(micro) score : {f1_score(y_test,model.predict(x_test),average='micro')}")

