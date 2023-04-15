from sklearn.datasets import load_iris, load_breast_cancer,fetch_covtype,load_digits,load_wine
import numpy as np
import random
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler,MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Input
from tensorflow.keras.callbacks import EarlyStopping as es
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
# 1. data prepare
# datasets=load_breast_cancer()
# x=datasets.data
# y=datasets.target

datalist= [load_iris, load_breast_cancer,fetch_covtype,load_digits,load_wine]
for i in datalist:
    x,y=i(return_X_y=True)
    # x,y=load_iris(return_X_y=True)
    # x,y=fetch_california_housing(return_X_y=True)


    # encoder=OneHotEncoder()
    # y=encoder.fit_transform(np.reshape(y,(len(y),1)))
    # print(y.shape)
    try:
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
        scaler=RobustScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model=Sequential()
        model.add(Input(shape=x.shape[1:]))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(3,activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
        model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test)
                ,callbacks=es(monitor='val_acc',mode='max',restore_best_weights=True,patience=20))
        results=model.evaluate(x_test,y_test)
        print(f'DNN acc : {results[1]}')
    except:
        x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
        scaler=RobustScaler()
        x_train=scaler.fit_transform(x_train)
        x_test=scaler.transform(x_test)
        model=Sequential()
        model.add(Input(shape=x.shape[1:]))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(256,activation='swish'))
        model.add(Dense(1,activation='linear'))
        model.compile(loss='mse',optimizer='adam')
        model.fit(x_train,y_train,epochs=1000,batch_size=len(x_train)//99,validation_data=(x_test,y_test)
                ,callbacks=es(monitor='val_loss',mode='min',restore_best_weights=True,patience=20))
        y_pred=model.predict(x_test)
        from sklearn.metrics import r2_score
        print(f'DNN결정계수 : {r2_score(y_test,y_pred)}')


    # 4. evaluate,predict


    # 5. SVM(support vector machine)
    from sklearn.svm import LinearSVC
    model=LinearSVC(penalty='l2')
    try:
        model.fit(x_train,y_train)
        print(f'Support Vector Machine : {model.score(x_test,y_test)}')
    except:
        print('Support Vector Machine 사용 불가능')



    from sklearn.linear_model import LogisticRegression
    model=LogisticRegression()
    try:
        model.fit(x_train,y_train)
        print(f'Logistic Regression : {model.score(x_test,y_test)}')
    except:
        print(f'Logistic Regression 사용 불가능')

    from sklearn.tree import DecisionTreeClassifier
    model=DecisionTreeClassifier()
    try:
        model.fit(x_train,y_train)
        print(f'Decision Tree Classifier : {model.score(x_test,y_test)}')
    except:
        print(f'Decision Trees Classifier 사용 불가능')

    from sklearn.tree import DecisionTreeRegressor
    mode=DecisionTreeRegressor()
    try:
        model.fit(x_train,y_train)
        print(f'Decision Tree Regressor : {model.score(x_test,y_test)}')
    except:
        print('Decision Tree Regressor 사용 불가능')
        
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier(random_state=seed)
    try:
        model.fit(x_train,y_train)
        print(f'Random Forest Classifier : {model.score(x_test,y_test)}')
    except:
        print('Random Forest Classifier 사용 불가능')

    from sklearn.ensemble import RandomForestRegressor
    model=RandomForestRegressor(random_state=seed)
    try:
        model.fit(x_train,y_train)
        print(f'Random forest Regressor : {model.score(x_test,y_test)}')
    except:
        print('Random forest Regressor 사용 불가능')