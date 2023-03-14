import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random, time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dense,Input
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. model build
input1=Input(shape=(x.shape[1],))
Dense1=Dense(16,activation='relu')(input1)
Dense2=Dense(16,activation='relu')(Dense1)
Dense3=Dense(16,activation='relu')(Dense2)
Dense4=Dense(16,activation='relu')(Dense3)
output1=Dense(1)(Dense4)
model=Model(inputs=input1,outputs=output1)
model.summary()

# model.save('./_save/keras26_1_save_model.h5')
model.save_weights('./_save/keras26_5_save_weights1.h5')

# 3. compile, training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,epochs=10000
          ,batch_size=len(x),validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=500,verbose=True,restore_best_weights=True))

# model.save('./_save/keras26_3_save_model.h5')
model.save_weights('./_save/keras26_5_save_weights2.h5')

# 4. predict,evaluate
print(f'loss : {model.evaluate(x_test,y_test)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')