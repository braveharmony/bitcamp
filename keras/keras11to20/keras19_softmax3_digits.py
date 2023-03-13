import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=load_digits()
x=datasets.data
y=datasets.target

# print(x.shape,y.shape)
print(np.unique(y))
y=pd.Categorical(y)
y=np.array(pd.get_dummies(y,prefix='number'))
# print(x.shape,y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True,stratify=y)

# 2. model build
model=Sequential()
model.add(Dense(64,input_dim=x.shape[1],activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=True,restore_best_weights=True)
hist=model.fit(x_train,y_train,batch_size=len(x),validation_split=0.2,verbose=True
            #    ,callbacks=es
               ,epochs=500)

# 4. predict,evaluate
y_predict=np.array(model.predict(x_test))
y_predict=np.argmax(y_predict,axis=1)
y_test=np.argmax(y_test,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['loss'],label='loss')
plt.title('loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()