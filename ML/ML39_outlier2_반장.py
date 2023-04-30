from typing import Union,Iterable
import numpy as np
import pandas as pd

##########################################################################################
def index_outliers(data_out:Union[np.ndarray, Iterable])->np.ndarray:
    data_out=np.array(data_out)
    quartile_1,quartile_3=np.percentile(data_out,[25,75])
    iqr=quartile_3-quartile_1
    lower_bound=quartile_1-1.5*iqr
    upper_bound=quartile_3+1.5*iqr
    return np.where((data_out<lower_bound)|(data_out>upper_bound))[0]
##########################################################################################
def index_2d_outliers(data_2d: Union[np.ndarray,pd.DataFrame]) -> np.ndarray:
    outlier_indices = []
    data_2d=np.array(data_2d)
    for idx, row in enumerate(data_2d):
        outliers = index_outliers(row)
        if outliers.size:
            outlier_indices+=[[idx, outlier] for outlier in outliers]
    return np.array(outlier_indices)
##########################################################################################
def remove_outliers(data:Union[np.ndarray, Iterable])->np.ndarray:
    data=np.array(data)
    outlier = index_outliers(data)
    if len(outlier) == 0:
        return data
    else:
        data_clean = data.astype(np.float64)
        data_clean[outlier] = np.nan
        return data_clean
##########################################################################################
def remove_2d_outliers(data_2d: Union[np.ndarray,pd.DataFrame]) -> np.ndarray:
    data_2d = np.array(data_2d).astype(np.float64).T
    outlier_indices = index_2d_outliers(data_2d)
    if len(outlier_indices) == 0:
        return data_2d.T
    for row, col in outlier_indices:
        data_2d[row, col] = np.nan
    return data_2d.T
##########################################################################################
def find_outliers(data:Union[np.ndarray, Iterable])->np.ndarray:
    outlier_data = np.full(len(data),np.nan)
    outlier = index_outliers(data)
    for i in outlier:
        outlier_data[i]=data[i]
    return outlier_data
##########################################################################################
def find_2d_outliers(data_2d: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    outlier_data = []
    for row in np.array(data_2d).T:
        outliers = find_outliers(row)
        if outliers.size:
            outlier_data += [outliers]
    return np.array(outlier_data).T
##########################################################################################
# aaa=[[2,3,4,-10,5,6,7,8,9,10,11,12,50]
#      ,[100,200,-30,400,500,600,-70000,800,900,100,210,420,350]]
# print(index_2d_outliers(aaa))
# print(remove_2d_outliers(aaa))
# print(find_2d_outliers(aaa))
# import matplotlib.pyplot as plt
# plt.boxplot(aaa)
# plt.show()