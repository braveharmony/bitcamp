import numpy as np
import pandas as pd
data=pd.DataFrame([[2,None,6,8,10],
                   [2,4,None,8,None],
                   [2,4,6,8,10],
                   [None,4,None,8,None]]).T
data.columns=[f'x{i}'for i in range(1,5)]

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer,KNNImputer,IterativeImputer
from xgboost import XGBRegressor
imputer=SimpleImputer(strategy='median')
print(imputer.fit_transform(data))
imputer=KNNImputer()
print(imputer.fit_transform(data))
imputer=IterativeImputer(estimator=XGBRegressor())
print(imputer.fit_transform(data))
