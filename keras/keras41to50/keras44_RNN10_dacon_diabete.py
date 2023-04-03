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

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
df = pd.read_csv('./_data/kaggle_bike/train.csv',index_col=0)
dft = pd.read_csv('./_data/kaggle_bike/test.csv',index_col=0)
dfs = pd.read_csv('./_data/kaggle_bike/sampleSubmission.csv')

df=df.dropna()
x=df.drop(df.columns[-1],axis=1)
y=df[df.columns[-1]]
print(x.columns)
print(dft.columns)


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

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
dft=reshape(dft)


print(x_train.shape,x_test.shape,dft.shape)

# 2. model build
model=Sequential(SimpleRNN(16,activation='linear',input_shape=x_train.shape[1:]))
model.add(Dense(1,activation='sigmoid'))

# 3. compile,training
model.compile(loss='binary_crossentropy',optimizer='adam')
model.fit(x_train,y_train,epochs=100,batch_size=len(x)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',restore_best_weights=True,verbose=True,patience=5))

# 4. predict,evaluate
def RMSE(y_test,y_pred):
    from sklearn.metrics import mean_squared_error
    return np.sqrt(mean_squared_error(y_test,y_pred))
y_pred=np.round(model.predict(x_test))
print(f'RMSE : {RMSE(y_test,y_pred)}\n결정계수 : {r2_score(y_test,y_pred)}')

# 5. save
y_predict=np.round(model.predict(dft))
dfs[df.columns[-1]]=y_predict
now=datetime.datetime.now().strftime('%H시%M분')
dfs.to_csv('')
dfs.to_csv(f'./_save/dacon_diabete/03_23/{now}_forsub.csv',index=False)