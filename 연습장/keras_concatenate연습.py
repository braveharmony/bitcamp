import tensorflow as tf
import numpy as np
import pandas as pd
import random
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Input,concatenate,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

datasets=fetch_california_housing()
# print(datasets)
# print(datasets.DESCR)
data=datasets.data
target=datasets.target
feature_names=datasets.feature_names
# print(data.shape,target.shape)
# print(feature_names)

# # plt.figure(1)
# for i in range(len(feature_names)):
#     plt.figure(str(i)+' '+feature_names[i])
#     # plt.subplot(3,int(len(feature_names)//3)+1,i+1)
#     plt.scatter(data.T[i],target,s=1,c="green")
#     plt.xlabel(str(i)+' '+feature_names[i],fontsize=5)
#     plt.ylabel('target',fontsize=5)
# plt.show()

data_train,data_test,y_train,y_test=train_test_split(data,target,train_size=0.7,random_state=seed)

x0_train=data_train.T[0].T
x1_train=data_train.T[1:6].T
x2_train=data_train.T[6:len(feature_names)].T

x0_test=data_test.T[0].T
x1_test=data_test.T[1:6].T
x2_test=data_test.T[6:len(feature_names)].T

print(x0_train.shape,x1_train.shape,x2_train.shape)

# 0<-linear 1~5<-모르겠음 6~7<-xor???
input0=Input(shape=(1,))
input1=Input(shape=(5,))
input2=Input(shape=(2,))

# 0layer, linear regration
nn1=Dense(10,activation='linear')(input0)
nn1=Dense(10,activation='linear')(nn1)

# 1~5layer, deeplearning
nn2=Dense(64,activation='relu')(input1)
nn2=Dropout(0.25)(nn2)
nn2=Dense(64,activation='relu')(nn2)

# 6~7layer, deeplearning
nn3=Dense(64,activation=LeakyReLU())(input2)
nn3=Dropout(0.25)(nn3)
nn3=Dense(64,activation=LeakyReLU())(nn3)

# merge
merge=concatenate([nn1,nn2,nn3])

# output
output=Dense(1,activation="linear")(merge)

# Model
model=Model(inputs=[input0,input1,input2],outputs=output)

# compile,train
model.compile(loss='mse',optimizer='adam')
model.fit([x0_train,x1_train,x2_train],y_train,batch_size=2000,epochs=1000,validation_split=0.1)

# evaluate,compile
loss=model.evaluate([x0_test,x1_test,x2_test],y_test)
print(f'loss : {loss}')
y_predict=model.predict([x0_test,x1_test,x2_test],)
print(f'r2Score : {r2_score(y_test,y_predict)}')