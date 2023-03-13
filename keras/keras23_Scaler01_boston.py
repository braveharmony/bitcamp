from sklearn.datasets import load_boston
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler # 일요일날 같이 보내기!
from matplotlib import pyplot as plt
import random,time
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets = load_boston()
x=datasets['data']
y=datasets['target']
feat=datasets['feature_names']
print(datasets.DESCR)
scaler=MinMaxScaler()
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

# 2. model build
model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))

starttime=time.time()
# 3. compile training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10,batch_size=len(x))
runtime=time.time()-starttime

# 4. evaluate,predict
loss= model.evaluate(x_test,y_test)
print(f'loss : {loss}')
print(f'runtime : {runtime}')

xt=x.T
for i in range(len(xt)):
    plt.subplot(3,5,i+1)
    plt.plot(xt[i],label=feat[i])
    plt.legend()
plt.show()