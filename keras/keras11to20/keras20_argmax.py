import tensorflow as tf
import numpy as np

a = np.array([[1,2,3],[6,4,5],[7,9,2],[3,2,1],[2,3,1]])
print(a.shape)
print(np.argmax(a))
print(np.argmax(a,axis=0)) # 행끼리 비교
print(np.argmax(a,axis=1)) # 열끼리 비교
print(np.argmax(a,axis=-1)) # 열끼리 비교
print(np.argmax(a,axis=-2)) # 행끼리 비교
