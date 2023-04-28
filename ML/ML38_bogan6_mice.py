import numpy as np
import pandas as pd
data=pd.DataFrame([[2,None,6,8,10],
                   [2,4,None,8,None],
                   [2,4,6,8,10],
                   [None,4,None,8,None]]).T
data.columns=[f'x{i}'for i in range(1,5)]

from impyute.imputation.cs import mice
impyute_df=mice(data.values) 
print(impyute_df)
impyute_df=pd.DataFrame(impyute_df)
impyute_df.columns=data.columns
print(impyute_df)

