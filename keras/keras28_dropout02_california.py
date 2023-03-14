import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing

# 0. seed initialization
seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# 2. model build
input1 = Input(shape=(x.shape[1],))
layer = Dense(30,activation='relu')(input1)
layer = Dropout(0.2)(layer)
layer = Dense(30,activation='relu')(layer)
layer = Dropout(0.2)(layer)
layer = Dense(30,activation='relu')(layer)
layer = Dropout(0.2)(layer)
layer = Dense(30,activation='relu')(layer)
layer = Dense(1)(layer)
model=Model(inputs=input1,outputs=layer)
# model.summary()


# 3. compile, training
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=10000
          ,verbose=True,validation_split=0.2,batch_size=len(x)
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
print(f'loss : {model.evaluate(x_test,y_test)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')
import matplotlib.pyplot as plt

plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')

plt.legend()
plt.show()