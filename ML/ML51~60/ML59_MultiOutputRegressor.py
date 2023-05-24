import numpy as np
import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error


x,y=load_linnerud(return_X_y=True)

from sklearn.linear_model import Lasso
model=Lasso()
model.fit(x,y)
print(type(model).__name__)
print(round(model.score(x,y),4))
print(round(mean_squared_error(y,model.predict(x)),4))

from sklearn.linear_model import Ridge
model=Ridge()
model.fit(x,y)
print(type(model).__name__)
print(round(model.score(x,y),4))
print(round(mean_squared_error(y,model.predict(x)),4))

from xgboost import XGBRegressor
model=XGBRegressor()
model.fit(x,y)
print(type(model).__name__)
print(round(model.score(x,y),4))
print(round(mean_squared_error(y,model.predict(x)),4))

from catboost import CatBoostRegressor
model=CatBoostRegressor(verbose=False,loss_function='MultiRMSE')
model.fit(x,y)
print(type(model).__name__)
print(round(model.score(x,y),4))
print(round(mean_squared_error(y,model.predict(x)),4))

from lightgbm import LGBMRegressor
model=MultiOutputRegressor(LGBMRegressor())
model.fit(x,y)
print(type(model).__name__)
print(round(model.score(x,y),4))
print(round(mean_squared_error(y,model.predict(x)),4))
