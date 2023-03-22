# 1. data prepare
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target
print(f'x.shape : {x.shape} y.shape : {y.shape}')

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=0)
print(f'x_train.shape : {x_train.shape} x_test.shape :{x_test.shape}')
print(f'y_train.shape : {y_train.shape} y_test.shape : {y_test.shape}')
