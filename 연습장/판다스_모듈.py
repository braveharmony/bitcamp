


# 1. data prepare
import numpy as np
import pandas as pd
import random
import tensorflow as tf
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

df=pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0)
print(df)
print(df.isnull().sum())
print(df.info())

x=df.drop(df.columns[-1],axis=1)
y=df[df.columns[-1]]
print(x.shape,y.shape)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)

# 2. Model Build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU

model=Sequential()
model.add(Dense(16,input_dim=x.shape[1],activation="linear"))
model.add(Dense(8,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(8,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(1))

# 3. Compile,Training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=100,epochs=50,verbose=True)

# 4. Evaluation,Predict
from sklearn.metrics import r2_score
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
print(f'결정 계수 : {r2_score(y_test,y_predict)}')

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.scatter(range(y_test.shape[0]),y_predict,s=10,c='red',label='y_predict data')
plt.scatter(range(y_test.shape[0]),y_test,s=10,c='green',label='y_test data')
plt.legend()
plt.show()