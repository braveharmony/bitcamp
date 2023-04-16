import numpy as np
import random
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
import warnings
from sklearn.utils import all_estimators

warnings.filterwarnings('ignore')
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)

# 1. data prepare
x,y=load_digits(return_X_y=True)
x=x[:5000]
y=y[:5000]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. RandomForest
from sklearn.ensemble import RandomForestRegressor
# model=RandomForestRegressor(n_jobs=4,n_estimators=1000)
# allAlgorithms=all_estimators(type_filter='classifier')
i=0
for index,model in all_estimators(type_filter='classifier'):
    i+=1
    try :
        model=model()
        model.fit(x_train,y_train)
        print(f'{i}번 index : {index}')
        print(f'결정계수 : {model.score(x_test,y_test)}')
    except Exception as e : print(f'{i}번 index : {index}\n실패\nerror:{e}')
    print('=====================================================')
# 3. DNN
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Input,Dropout
from tensorflow.keras.callbacks import EarlyStopping as es
model=Sequential()
model.add(Input(shape=x.shape[1:]))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(len(np.unique(y)),activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_train,y_train,epochs=1000,validation_data=(x_test,y_test),batch_size=len(x_train)//50,callbacks=es(monitor='val_loss',mode='min'
                                                                                                               ,patience=50,restore_best_weights=True))
model.evaluate(x_train,y_train)
