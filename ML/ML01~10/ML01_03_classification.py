from sklearn.datasets import load_iris, load_breast_cancer,fetch_covtype,load_digits,load_wine
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping as es
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def do_models(x,y):
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
        # 2. model build
    model=Sequential()
    model.add(Input(shape=x.shape[1:]))
    model.add(Dense(256,activation='swish'))
    model.add(Dense(256,activation='swish'))
    model.add(Dense(256,activation='swish'))
    model.add(Dense(256,activation='swish'))
    model.add(Dense(256,activation='swish'))
    model.add(Dense(len(np.unique(y)),activation='softmax'))
    # 3. compile,training
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
    model.fit(x_train,y_train,epochs=1000,batch_size=len(x_train)//99,validation_data=(x_test,y_test)
            ,callbacks=es(monitor='val_acc',mode='max',restore_best_weights=True,patience=20))
    # 4. evaluate,predict
    results=model.evaluate(x_test,y_test)
    print(f'DNN acc : {results[1]}')
    model_list=[LinearSVC(penalty='l2'),LogisticRegression(),DecisionTreeClassifier(),DecisionTreeRegressor()
                ,RandomForestClassifier(random_state=seed),RandomForestRegressor(random_state=seed)]
    def runmodel(model,x_train,x_test,y_train,y_test):
        try:
            model.fit(x_train,y_train)
            print(f'{type(model).__name__} : {model.score(x_test,y_test)}')
        except:
            print(f'{type(model).__name__} 사용 불가능')
    for model in model_list:
        runmodel(model,x_train,x_test,y_train,y_test)


pd_path=['./_data/'+i for i in ['dacon_diabete/']]
# sklearn_datalist=[]
sklearn_datalist= [load_iris, load_breast_cancer,load_digits,load_wine]
for data in ['sklearn','pandas']:
    if data=='sklearn':
        for i in sklearn_datalist:
            # 1. data prepare
            x,y=i(return_X_y=True)
            do_models(x,y)
            print(f'{i.__name__} is finished')
    else :
        for i in pd_path:
            datasets=pd.read_csv(i+'train.csv',index_col=0)
            x=datasets.drop(datasets.columns[-1],axis=1)
            y=datasets[datasets.columns[-1]]
            do_models(np.array(x),np.array(y))
            print(f'{i} is finished')
