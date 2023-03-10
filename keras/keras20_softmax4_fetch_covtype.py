import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import random
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
datasets=fetch_covtype()
x=datasets.data
y=datasets.target
print(x.shape,y.shape)
print(np.unique(y))

y=np.array(pd.get_dummies(y))
print(y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,stratify=y)

# 2. model build
model=Sequential()
model.add(Dense(64,input_dim=x.shape[1],activation=LeakyReLU(0.8)))
model.add(Dense(32,activation=LeakyReLU(0.8)))
model.add(Dense(64,activation=LeakyReLU(0.8)))
model.add(Dense(32,activation=LeakyReLU(0.8)))
model.add(Dense(64,activation=LeakyReLU(0.8)))
model.add(Dense(y.shape[1],activation='softmax'))

# 3. compile,training
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_acc',mode='max',patience=200,restore_best_weights=True)
hist=model.fit(x_train,y_train,batch_size=len(x_train)//5,verbose=True,validation_split=0.2,callbacks=es,epochs=1000)

# 4. evaluate,predict
y_predict=model.predict(x_test)
y_test=np.argmax(y_test,axis=1)
y_predict=np.argmax(y_predict,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')

# 5. ploting
plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.title('val_loss')
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()