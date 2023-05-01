import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from imblearn.over_sampling import SMOTE

#1.Data prepare
datasets = load_wine()
x = pd.DataFrame(datasets.data)
y= pd.Series(datasets.target)
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

