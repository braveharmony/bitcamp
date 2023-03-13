import numpy as np
import tensorflow as tf
import pandas as pd 
import matplotlib as mpl
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(0)
np.random.seed(0)
tf.random.set_seed(seed)

# 1. data prepare
datasets = load_iris()
print(datasets.DESCR)
x=datasets.data
y=datasets.target
y=np.array(pd.get_dummies(y,prefix='number'))


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True,stratify=y)
x_test=MinMaxScaler().fit(x_train).transform(x_test)
x_train=MinMaxScaler().fit(x_train).transform(x_train)


# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True)
hist=model.fit(x_train,y_train,batch_size=len(x_train)
               ,callbacks=es
               ,validation_split=0.2,verbose=True,epochs=1000)

# 4. predict evaluate
y_predict=np.argmax(np.array(model.predict(x_test)),axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')

# 5. plot
plt.subplot(1,2,1)
plt.plot(hist.history['val_acc'],label='validation accuracy')
plt.subplot(1,2,2)
plt.plot(hist.history['val_loss'],label='validation loss')
plt.legend()
plt.show()