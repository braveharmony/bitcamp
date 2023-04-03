import numpy as np
import tensorflow as tf
import pandas as pd
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,SimpleRNN,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_iris

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets = load_iris()
x=datasets.data
y=datasets.target

y=np.array(pd.get_dummies(y,prefix='number'))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# def split_to_time(dataset,timesteps):
#     gen=(dataset[i:i+timesteps] for i in range(len(dataset)-timesteps+1))
#     return np.array(list(gen))
# timesteps=5
# x_train=split_to_time(x_train,timesteps)
# x_test=split_to_time(x_test,timesteps)
# dft=split_to_time(dft,timesteps)
# y_test=y_test[timesteps-1:]

def reshape(x):
    return np.reshape(x,list(x.shape)+[1])
x_train=reshape(x_train)
x_test=reshape(x_test)


print(x_train.shape,x_test.shape)

# 2. model build
model=Sequential(SimpleRNN(16,activation='linear',input_shape=x_train.shape[1:]))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=len(x)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=True,patience=5))

# 4. predict,evaluate
def RMSE(y_test,y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred=np.argmax(model.predict(x_test),axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'RMSE : {RMSE(y_test,y_pred)}\n결정계수 : {r2_score(y_test,y_pred)}')
