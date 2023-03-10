import tensorflow as tf
import random,numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data
path='./_data/kaggle_bike/'
df=pd.read_csv(path+'train.csv')
# print(df.describe)
# print(df.isnull().sum())
# print(df.columns)
dft=pd.read_csv(path+'test.csv')
# print(dft.columns)

x=df.drop([df.columns[0],df.columns[-1],df.columns[-2],df.columns[-3]],axis=1)
y=df[df.columns[-1]]
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)

# # 2. model build
# model=Sequential()
# model.add(Dense(1,input_dim=x.shape[1]))
# model.add(Dense(1))

# # 3. compile,training
# model.compile(loss='mse',optimizer='adam')
# model.fit(x_train,y_train,batch_size=len(x),epochs=1)

# # 4. evaluate predict
# loss=model.evaluate(x_test,y_test)
# print(f'loss : {loss}')

# y_predict=model.predict(x_test)
# def RMSE(y_test,y_predict):
#     return np.sqrt(mean_squared_error(y_test,y_predict))
# rmse=RMSE(y_test,y_predict)
# print(f'rmse : {rmse}')

# ################# dft로 가져와서출력 y만듬#############
# # print(dft)
# x=dft.drop([dft.columns[0]],axis=1)
# # print(x)
# y=model.predict(x)
# # print(y)

# ########### sampleSubmission에서 뜯어와서 y넣고 등록########
# dfsub=pd.read_csv(path+'sampleSubmission.csv')
# dfsub[df.columns[-1]]=y
# # print(dfsub)

# dfsub.to_csv('./_save/kaggle_bike/mltestsample2.csv',index=False)

# 번외 플로팅
plt.figure(input)
for i in range(len(x.columns)):
    plt.subplot(3,3,i+1)
    plt.scatter(x[x.columns[i]],y,s=5)
    plt.title(x.columns[i],fontsize=5)
plt.show()