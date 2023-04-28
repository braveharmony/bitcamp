# 과제
# 판다스->넘파이
# 넘파이->판다스
# 리스트->넘파이
# 리스트->판다스
import numpy as np
import pandas as pd
from typing import List

def pandas_to_numpy_2D(pandas:pd.DataFrame)->np.ndarray:
    return pandas.values

def numpy_to_pandas(numpy:np.ndarray,columns:List=[])->pd.DataFrame:
    if type(columns)!=list:
        columns=list(columns)
    if len(columns)!=numpy.shape[1]:
        columns=list(range(numpy.shape[1]))
    return pd.DataFrame(numpy,columns=columns)

def list_2D_to_numpy(list_2D: List) -> np.ndarray:
    assert all(isinstance(sublist, list) for sublist in list_2D)
    return np.array(list_2D)

def list_2D_to_pandas(list_2D: List, columns: List = []) -> pd.DataFrame:
    assert all(isinstance(sublist, list) for sublist in list_2D)
    numpy_array = list_2D_to_numpy(list_2D)
    if type(columns) != list:
        columns = list(columns)
    if len(columns) != numpy_array.shape[1]:
        columns = list(range(numpy_array.shape[1]))
    return pd.DataFrame(numpy_array, columns=columns)

