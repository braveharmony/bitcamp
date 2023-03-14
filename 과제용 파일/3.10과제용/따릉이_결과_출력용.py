import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
for i in range(10):
    # 0. seed initialization
    seed=i
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. data prepare
    path='./_data/DDarung/'
    df=pd.read_csv(path+'train.csv',index_col=0)
    dft=pd.read_csv(path+'test.csv',index_col=0)
    dfs=pd.read_csv(path+'submission.csv')
    df=df.dropna()
    # print(df)
    # print(df.describe())
    # print(df.info())
    # print(df.isnull().sum())
    x=df.drop([df.columns[-1]],axis=1)
    # print(x.describe)
    y=df[df.columns[-1]]
    print(df.columns)
    print(x.shape,y.shape)

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed)
    scaler=MinMaxScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    dft=scaler.transform(dft)

    # 2. model build
    model=Sequential()
    model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.55)))
    model.add(Dense(64,activation=LeakyReLU(0.55)))
    model.add(Dense(32,activation=LeakyReLU(0.55)))
    model.add(Dense(64,activation=LeakyReLU(0.55)))
    model.add(Dense(1))

    # 3. compile, training
    model.compile(loss='mse',optimizer='adam')
    es=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True)
    hist=model.fit(x_train,y_train,batch_size=len(x),validation_split=0.2,verbose=True,epochs=5000,callbacks=es)

    # 4. predict,evaluation
    def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
    print(f'RMSE : {RMSE(y_test,model.predict(x_test))}')

    # 5. save
    y_predict=model.predict(dft)
    dfs[df.columns[-1]]=y_predict
    pathsave='./_save/DDarung/03_13/'
    dfs.to_csv(pathsave+f'forsub{i}.csv',index=False)

# # 6. plot
# plt.subplot(1,2,1)
# plt.plot(hist.history['val_loss'],label='val_loss')
# plt.plot(hist.history['loss'],label='loss')
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(hist.history['val_loss'],label='val_loss')
# plt.legend()
# plt.show()
