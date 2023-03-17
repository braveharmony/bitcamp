import tensorflow as tf
import pandas as pd
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
for j in range(10):
    # 0. seed initialization
    seed=j
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
    for i in df.columns:
        if type(df[i][1])==type(str()):
            dft=dft.drop([i],axis=1)
            df=df.drop([i],axis=1)
    ddel=['Alley','PoolQC','Fence','MiscFeature','FireplaceQu']
    for i in ddel:
        dft=dft.drop([i],axis=1)
        df=df.drop([i],axis=1)
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

    # 2. model build
    model=Sequential()
    model.add(Dense(64,input_dim=x.shape[1],activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(64,activation='relu'))
    model.add(Dense(1))

    # 3. compile,fit
    model.compile(loss='mse',optimizer='adam')
    model.fit(x,y,batch_size=len(x),epochs=10000,validation_split=0.2,
            callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=600,restore_best_weights=True))

    # 4. evaluate,predict
    y_predict=np.array(model.predict(dft))
    def renonemean(arr):
        arr = np.array(arr)
        arr = np.where(arr == None, np.nan, arr)
        mean = np.nanmean(arr)
        arr = np.where(np.isnan(arr), mean, arr)
        return arr
    dfs[dfs.columns[-1]]=renonemean(y_predict)
    dfs.to_csv(f'./_save/kaggle_house/03_17/forsub{j}.csv',index=False)