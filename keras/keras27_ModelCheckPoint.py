import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
# 0. seed initialization
seed=1
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_boston()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
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
from tensorflow.python.keras.callbacks import ModelCheckpoint
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,epochs=10000
          ,batch_size=len(x_train)//1,validation_split=0.2
          ,callbacks=(EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True,verbose=True),
                        ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,save_best_only=True,filepath='./_save/MCP/keras27_ModelCheckPoint.hdf5')))

# 4. predict,evaluate
print(f'loss :  {model.evaluate(x_test,y_test)}\n결정계수 : {r2_score(y_test,model.predict(x_test))}')

# 5. history log
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.legend()
plt.show()

# patience 500
# loss :  7.91240119934082
# 결정계수 : 0.9199371527028835