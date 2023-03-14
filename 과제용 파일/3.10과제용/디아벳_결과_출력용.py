import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,MaxAbsScaler,RobustScaler,StandardScaler
from matplotlib import pyplot as plt

for i in range(10):
    # 0. seed initialization
    seed=i
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. data prepare
    path='./_data/dacon_diabete/'
    df=pd.read_csv(path+'train.csv',index_col=0)
    dft=pd.read_csv(path+'test.csv',index_col=0)
    dfs=pd.read_csv(path+'sample_submission.csv')
    x=df.drop([df.columns[-1]],axis=1)
    y=df[df.columns[-1]]

    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
    scaler=MinMaxScaler()
    # scaler=MaxAbsScaler()
    scaler.fit(x_train)
    x_train=scaler.transform(x_train)
    x_test=scaler.transform(x_test)
    dft=scaler.transform(dft)

    # 2. model build
    model=Sequential()
    model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.75)))
    model.add(Dense(16,activation=LeakyReLU(0.75)))
    model.add(Dense(32,activation=LeakyReLU(0.75)))
    model.add(Dense(16,activation=LeakyReLU(0.75)))
    model.add(Dense(32,activation=LeakyReLU(0.75)))
    model.add(Dense(16,activation=LeakyReLU(0.75)))
    model.add(Dense(1,activation='sigmoid'))

    # 3. compile,training
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
    es=EarlyStopping(monitor='val_acc',mode='max',patience=100,restore_best_weights=True)
    hist=model.fit(x_train,y_train,batch_size=len(x_train),validation_split=0.2,verbose=True,callbacks=es,epochs=1000)

    # 4. evaluate,predict
    print(f'accuracy : {accuracy_score(y_test,np.round(model.predict(x_test)))}')
    y_predict=np.round(model.predict(dft))
    dfs[df.columns[-1]]=y_predict
    pathsave='./_save/dacon_diabete/03_13/'
    dfs.to_csv(pathsave+f'forsub{i}.csv',index=False)

# # 6. plot
# plt.subplot(1,2,1)
# plt.plot(hist.history['loss'],label="loss")
# plt.plot(hist.history['val_loss'],label="val_loss")
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(hist.history['val_acc'],label='val_acc')
# plt.legend()
# plt.show()