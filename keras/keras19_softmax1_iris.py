import tensorflow as tf
import numpy as np
import pandas as pd
import random
import matplotlib as mpl
from keras.utils import to_categorical
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)
x=datasets.data
y=datasets.target
print(y.shape)
y1=list()
y2=list()
y3=list()
print(f'y의 라벨값 : {np.unique(y)}')

# one hot incoding
def one_hot(y):
    for i in y:
        if i==0:
            y1.append(1)
            y2.append(0)
            y3.append(0)
        elif i==1:
            y1.append(0)
            y2.append(1)
            y3.append(0)
        else:
            y1.append(0)
            y2.append(0)
            y3.append(1)
    return np.array([y1,y2,y3]).T
# y=one_hot(y)
y=to_categorical(y)
# y = pd.Categorical(y)
# y = np.array(pd.get_dummies(y, prefix='number'))

# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,train_size=0.9,random_state=seed,stratify=y)
print(y_test)

# 2. 모델 구성
model = Sequential()
model.add(Dense(50, activation='relu', input_dim=x.shape[1]))
model.add(Dense(40, activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))



# 3. compile, training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=20,restore_best_weights=True,verbose=True)
hist=model.fit(x_train,y_train,epochs=100,batch_size=len(x_train),
          validation_split=0.2,
          verbose=1)

# 4. evaluate predict
result=model.evaluate(x_test,y_test)
y_predict=model.predict(x_test)
y_predict=np.array(y_predict)
y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)

acc=accuracy_score(y_test,y_predict)
print('accurancy : ',acc)

import matplotlib.pyplot as plt
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['loss'])
plt.show()