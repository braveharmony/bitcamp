from typing import Union,Iterable
import numpy as np
import pandas as pd

def outliers(data_out:np.ndarray)->np.ndarray:
    quartile_1,quartile_3=np.percentile(data_out,[25,75])
    iqr=quartile_3-quartile_1
    lower_bound=quartile_1-1.5*iqr
    upper_bound=quartile_3+1.5*iqr
    return np.where((data_out<lower_bound)|(data_out>upper_bound))[0]

def remove_outliers(data:Union[np.ndarray, Iterable])->np.ndarray:
    data=np.array(data)
    outlier = outliers(data)
    if len(outlier) == 0:
        return data
    else:
        data_clean = data.astype(np.float64)
        data_clean[outlier] = np.nan
        return data_clean
aaa=[-10,2,3,4,5,6,7,8,9,10,11,12,50]
print(remove_outliers(aaa))
series=pd.Series(aaa)
print(remove_outliers(series))
nparray=np.array(aaa)
print(remove_outliers(nparray))
tuple=(*aaa,)
print(remove_outliers(tuple))

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()