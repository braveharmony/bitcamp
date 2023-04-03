import tensorflow as tf
import pandas as pd
import numpy as np
import random,time,datetime
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout,Conv2D,Flatten,SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# 1. data prepare
path = './_data/kaggle_house/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
# print(df.info())
# print(dft.info())
# print(type(df[df.columns[2]][1]),type(str()))
le=LabelEncoder()
for i in dft.columns:
    if dft[i].dtypes=='object':
        dft[i]=le.fit_transform(dft[i])
        df[i]=le.fit_transform(df[i])
# print(df.info())
# print(dft.info())
# print(dfs.info())

df=df.dropna()
print(df.info())
# 아무튼 정제
x=df.drop([dfs.columns[-1]],axis=1)
y=df[dfs.columns[-1]]
print(x.shape,y.shape)
print(np.unique(y))
print(dfs.shape,dft.shape)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

def split_to_time(dataset,timesteps):
    gen=(dataset[i:i+timesteps] for i in range(len(dataset)-timesteps+1))
    return np.array(list(gen))
timesteps=1
x_train=split_to_time(x_train,timesteps)
x_test=split_to_time(x_test,timesteps)
dft=split_to_time(dft,timesteps)
y_test=y_test[timesteps-1:]

# def reshape(x):
#     return np.reshape(x,list(x.shape)+[1])
# x_train=reshape(x_train)
# x_test=reshape(x_test)
# dft=reshape(dft)


# 2. model build
model=Sequential(SimpleRNN(16,activation='linear',input_shape=x_train.shape[1:]))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(1))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=1,batch_size=100
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=True,patience=50))

# 4. predict,evaluate
def RMSE(y_test,y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred=model.predict(x_test)
print(f'RMSE : {RMSE(y_test,y_pred)}\n결정계수 : {r2_score(y_test,y_pred)}')

# 5. save
y_predict=model.predict(dft)
dfs[df.columns[-1]]=y_predict
now=datetime.datetime.now().strftime('%H시%M분')
dfs.to_csv('')
dfs.to_csv(f'./_save/kaggle_house/03_23/{now}_forsub.csv',index=False)