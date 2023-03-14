import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout
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

    # print(df.info(),dft.info())

    x=df.drop([df.columns[-1]],axis=1)
    y=df[[df.columns[-1]]]
    # print(x.info(),y.info())

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.85,random_state=seed,shuffle=True)
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    dft=scaler.transform(dft)

    # 2. model build
    input1=Input(shape=(x.shape[1],))
    layer = Dense(32,activation=LeakyReLU(0.55))(input1)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    layer = Dense(1)(layer)
    model=Model(inputs=input1,outputs=layer)
    model.summary()

    # 3. compile,training
    model.compile(loss='mse',optimizer='adam')
    hist = model.fit(x_train,y_train,epochs=10000
            ,batch_size=len(x),validation_split=0.2,verbose=True
            ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=True,restore_best_weights=True))

    # 4. predict,evaluate
    def RMSE(y_test,y_predict):
        return np.sqrt(mean_squared_error(y_test,y_predict))
    print(f"RMSE : {RMSE(y_test,model.predict(x_test))}")

    dfs[df.columns[-1]]=model.predict(dft)
    dfs.to_csv(f'./_save/DDarung/03_14/forsub{i}.csv',index=False)

    # plt.plot(hist.history['val_loss'],label='val_loss')
    # plt.plot(hist.history['loss'],label='loss')
    # plt.legend()
    # plt.show()