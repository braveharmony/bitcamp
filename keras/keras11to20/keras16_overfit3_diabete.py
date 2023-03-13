import numpy as np
import tensorflow as tf
import random
import pandas as pd
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_diabetes()
x=datasets['data']
y=datasets['target']
feature_names=datasets['feature_names']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
# 2. model build
model=Sequential()
model.add(Dense(10,input_dim=x.shape[1],activation="relu"))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))

# 3. compile,training
model.compile(loss='mse',optimizer='adam')
hist=model.fit(x_train,y_train,batch_size=len(x),verbose=True,validation_split=0.2,epochs=1000)

# 4. evaluate,predict
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict = model.predict(x_test)
print(f'r2 : {r2_score(y_test,y_predict)}')

# 5. plot 
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.',c='red',label='loss')
plt.plot(hist.history['val_loss'],marker='.',c='blue',label='val_loss')
plt.title('boston')
plt.xlabel('epochs')
plt.ylabel('loss and val_loss')
plt.legend()
plt.grid()
plt.show()
