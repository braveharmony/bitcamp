import tensorflow as tf
import numpy as np
import pandas as pd
import random,time
import matplotlib as mpl
from tensorflow.python.keras.models import Model,load_model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# 0. seed initialization
seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. model build
Dense1 = Input(shape=(x.shape[1],))
Dense2 = Dense(32, activation=LeakyReLU(0.7))(Dense1)
Dense3 = Dense(32, activation=LeakyReLU(0.7))(Dense2)
Dense4 = Dense(32, activation=LeakyReLU(0.7))(Dense3)
Dense5 = Dense(32, activation=LeakyReLU(0.7))(Dense4)
Dense6 = Dense(32, activation=LeakyReLU(0.7))(Dense5)
Dense7 = Dense(32, activation=LeakyReLU(0.7))(Dense6)
Dense8 = Dense(32, activation=LeakyReLU(0.7))(Dense7)
Dense9 = Dense(1)(Dense8)
model = Model(inputs=Dense1,outputs=Dense9)
model.summary()

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=10000
          ,batch_size=len(x),validation_split=0.2,verbose=True,
          callbacks=(EarlyStopping(monitor='val_loss',mode='min',
                                #    restore_best_weights=True,
                                   verbose=True,patience=50),
                     ModelCheckpoint(monitor='val_loss',mode='auto',save_best_only=True,verbose=True,filepath='./_save/MCP/keras27_3_MCP.hdf5')))


print('================= 1. 기본 출력 ====================')
model.save('./_save/MCP/keras27_3_save_models.h5')
print(f'loss : {model.evaluate(x_test,y_test,verbose=False)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')

print('================= 2. load_model 출력 ====================')
model=load_model('./_save/MCP/keras27_3_save_models.h5')
print(f'loss : {model.evaluate(x_test,y_test,verbose=False)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')

print('================= 3. MCP 출력 ====================')
model =load_model('./_save/MCP/keras27_3_MCP.hdf5')
print(f'loss : {model.evaluate(x_test,y_test,verbose=False)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')

# 4. history log
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()