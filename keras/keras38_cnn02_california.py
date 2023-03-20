import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Model,Sequential
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout,Conv2D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing

# 0. seed initialization
seed=2
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
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]//4,2,2))
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]//4,2,2))
print(x_train.shape,x_test.shape)



# 2. model build
model=Sequential()
model.add(Conv2D(input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
                 ,filters=32
                 ,kernel_size=(2,2)
                 ,padding='same'
                 ,activation='relu'))
model.add(Conv2D(filters=32
                 ,kernel_size=(2,2)
                 ,padding='valid'
                 ,activation='relu'))
model.add(Flatten())
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(1))
model.summary()


# 3. compile, training
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=2000
          ,verbose=True,validation_split=0.2,batch_size=len(x)
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True,verbose=True))

# 4. predict,evaluate
print(f'loss : {model.evaluate(x_test,y_test)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')


# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'],label='loss')
# plt.plot(hist.history['val_loss'],label='val_loss')

# plt.legend()
# plt.show()