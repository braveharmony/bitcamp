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
    path='./_data/kaggle_bike/'
    df=pd.read_csv(path+'train.csv')
    dft=pd.read_csv(path+'test.csv')
    dfs=pd.read_csv(path+'sampleSubmission.csv')
    # print(df.columns)
    # print(dft.columns)
    x=df.drop([df.columns[0],df.columns[-1],df.columns[-2],df.columns[-3]],axis=1)
    dft=dft.drop([df.columns[0]],axis=1)
    y=df[df.columns[-1]]


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


    # 3. compile,training
    model.compile(loss='mse',optimizer='adam')
    es=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True)
    hist=model.fit(x_train,y_train,batch_size=len(x_train),validation_split=0.2,verbose=True,callbacks=es,epochs=3000)


    # 4. evaluate,predict
    def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
    print(f'RMSE : {RMSE(y_test,model.predict(x_test))}')

    # 5. save
    y_predict=model.predict(dft)
    dfs[df.columns[-1]]=y_predict
    pathsave='./_save/kaggle_bike/03_13/'
    dfs.to_csv(pathsave+f'forsub{i}.csv',index=False)

# # 6. plot
# plt.plot(hist.history['loss'],label='loss')
# plt.plot(hist.history['val_loss'],label='val_loss')
# plt.legend()
# plt.show()