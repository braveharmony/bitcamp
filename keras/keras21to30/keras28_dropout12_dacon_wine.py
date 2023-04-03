import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,Dropout,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
for i in range(1,2):
    # 0. seed initialization
    seed=i
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # 1. data prepare
    path='./_data/dacon_wine/'
    df=pd.read_csv(path+'train.csv',index_col=0)
    dft=pd.read_csv(path+'test.csv',index_col=0)
    dfs=pd.read_csv(path+'sample_submission.csv')
    # print(df.columns,dft.columns)
    x=df.drop([df.columns[0],df.columns[-1]],axis=1)
    y=df[df.columns[0]]
    dft=dft.drop([df.columns[-1]],axis=1)
    # print(x.columns,dft.columns)
    print(np.unique(y))
    y=np.array(pd.get_dummies(y,prefix='value'))
    # print(y.shape)
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True,stratify=y)
    scaler=MinMaxScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    dft=scaler.transform(dft)
    
    # print(x_train.shape)
    # 2. model build
    input1=Input(shape=(x.shape[1],))
    layer = Dense(32,activation=LeakyReLU(0.55))(input1)
    layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    # layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    # layer = Dropout(0.1)(layer)
    layer = Dense(32,activation=LeakyReLU(0.55))(layer)
    layer = Dropout(0.1)(layer)
    # layer = Dense(32,activation=LeakyReLU(0.85))(layer)
    # layer = Dropout(0.1)(layer)
    # layer = Dense(32,activation=LeakyReLU(0.85))(layer)
    # layer = Dropout(0.1)(layer)
    # layer = Dense(32,activation=LeakyReLU(0.85))(layer)
    # layer = Dropout(0.1)(layer)
    # layer = Dense(32,activation=LeakyReLU(0.85))(layer)
    # layer = Dropout(0.1)(layer)
    layer = Dense(y.shape[1],activation='softmax')(layer)
    model=Model(inputs=input1,outputs=layer)
    # model.summary()
    # 3. compile,training
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    hist = model.fit(x_train,y_train,epochs=30000
            ,batch_size=len(x),validation_split=0.2,verbose=True
            ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True))

    # 4. predict, evaluate
    y_predict = np.argmax(model.predict(x_test),axis=1)+3
    y_test = np.argmax(y_test,axis=1)+3
    print(f'accuaracy : {accuracy_score(y_test,y_predict)}')
    # print(dfs.columns[0])
    # print(np.argmax(model.predict(dft),axis=1)+3)
    dfs[df.columns[0]] =np.argmax(model.predict(dft),axis=1)+3
    dfs.to_csv(f'./_save/dacon_wine/03_14/forsub{i}.csv',index=False)
    plt.subplot(1,2,1)
    plt.plot(hist.history['val_acc'],label='val_acc')
    plt.plot(hist.history['acc'],label='acc')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history['val_loss'],label='val_loss')
    plt.plot(hist.history['loss'],label='loss')
    plt.legend()
    plt.show()