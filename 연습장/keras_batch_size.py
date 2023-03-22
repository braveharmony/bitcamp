# 1. data prepare
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0,shuffle=True)
print(f'x_train.shape : {x_train.shape} x_test.shape :{x_test.shape}')
print(f'y_train.shape : {y_train.shape} y_test.shape : {y_test.shape}')

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))

# 3. compile
model.compile(loss='mse',optimizer='adam')

import time
from sklearn.metrics import r2_score
import numpy as np
starttime=time.time()
model.fit(x_train,y_train,epochs=1000,batch_size=len(x_train)//10)
print('1000번 학습')
print(f'runtime : {np.round(time.time()-starttime,2)} 초\n결정계수 : {r2_score(y_test,model.predict(x_test))}')
