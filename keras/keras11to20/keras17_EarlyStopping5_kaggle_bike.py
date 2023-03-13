import random
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 1. Data prepare
path='./_data/kaggle_bike/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sampleSubmission.csv')
# print(df.info())
# print(df.columns)
# print(dft.columns)
x=df.drop([df.columns[-3],df.columns[-2],df.columns[-1]],axis=1)
y=df[df.columns[-1]]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)


# 2. model build
model=Sequential()
model.add(Dense(1,input_dim=x.shape[1]))


# 3. compile, training
model.compile(loss='mse',optimizer='adam')
es=EarlyStopping(monitor='val_loss',mode='min',patience=10,verbose=True,restore_best_weights=True)
model.fit(x,y,batch_size=len(x_train)//4,verbose=True,validation_split=0.2,callbacks=es,epochs=1)


# 4. evaluate, save
print(f'loss : {model.evaluate(x_test,y_test,batch_size=len(x_test)//4)}')
y_predict=model.predict(dft)
dfs[df.columns[-1]]=y_predict
pathsave='./_save/kaggle_bike/'
dfs.to_csv(pathsave+'submission_03_08.csv',index=False)