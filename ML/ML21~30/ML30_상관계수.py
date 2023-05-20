import numpy as np
import pandas as pd
from sklearn.datasets import load_iris


dataset=load_iris()
print(dataset.feature_names)

x=dataset['data']
y=dataset['target']

df = pd.DataFrame(x,columns=dataset.feature_names)
print(df)
df['Target(Y)']=y
print(df.corr())