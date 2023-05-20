import numpy as np
import pandas as pd
from datetime import datetime

data=pd.DataFrame([[2,None,6,8,10],
                   [2,4,None,8,None],
                   [2,4,6,8,10],
                   [None,4,None,8,None]]).T
print(data)
data.columns=[f'x{i}'for i in range(1,5)]
print(data)

#0. 결측치 확인
print(data.isna())
print(data.isna().sum())
print(data.info())

print('=====================결측치 삭제=====================')
print(data.dropna())
print('=====================결측치 삭제=====================')
print(data.dropna(axis=0))
print('=====================결측치 삭제=====================')
print(data.dropna(axis=1))
print('=====================평균값 처리=====================')
print(data.fillna(data.mean()))
print('=====================중위값 처리=====================')
print(data.fillna(data.median()))
print('=====================이전값 처리=====================')
# print(data.ffill())
print(data.fillna(method='ffill'))
print('=====================이후값 처리=====================')
# print(data.bfill())
print(data.fillna(method='bfill'))
print('=====================선형값 처리=====================')
print(data.interpolate().fillna(method='bfill'))



print('=====================임의값 처리=====================')
data2=data.copy()
data2['x1']=data['x1'].fillna(data['x1'].mean())
data2['x2']=data['x2'].fillna(data['x2'].median())
data2['x4']=data['x4'].fillna(method='ffill')
data2['x4']=data2['x4'].fillna(777777)
print(data2)
del data2