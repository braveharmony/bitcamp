import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_covtype
import time
# 1. 데이터
datasets=fetch_covtype()
x=datasets['data']
y=datasets['target']

y=np.array(pd.get_dummies(y))

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size= 0.8,shuffle=True,stratify=y)

# 2. model build
model =Sequential()
model.add(Dense(50,input_dim=x.shape[-1],activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(7,activation='softmax'))
# model.summary
# sparse_categorical_crossentropy
start_time=time.time()

# 3. compile, build
model.compile(loss='categorical_crossentropy'
              ,optimizer='adam',metrics=['acc'])
model.fit(x_train,y_train,batch_size=len(x)//3,validation_split=0.2,verbose=True,epochs=10)
runtime=time.time()-start_time
print(f'걸린 시간 :{runtime//3600}시간 {runtime//60}분 {round((runtime*100)%60,2)}초')
# 4. predict,evaluate
print(f'accuracy : {accuracy_score(np.argmax(y_test,axis=1),np.argmax(model.predict(x_test),axis=1))}')