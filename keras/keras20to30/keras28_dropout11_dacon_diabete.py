import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')

print(df.columns,dft.columns)
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]


x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True,stratify=y)
scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 2. model build
input=Input(shape=(x.shape[1],))
layer=Dense(32,activation=LeakyReLU(0.5))(input)
layer=Dropout(0.05)(layer)
layer=Dense(32,activation=LeakyReLU(0.5))(layer)
layer=Dropout(0.05)(layer)
layer=Dense(32,activation=LeakyReLU(0.5))(layer)
layer=Dropout(0.05)(layer)
layer=Dense(32,activation=LeakyReLU(0.5))(layer)
layer=Dropout(0.05)(layer)
layer=Dense(32,activation=LeakyReLU(0.5))(layer)
layer=Dropout(0.05)(layer)
layer=Dense(32,activation=LeakyReLU(0.5))(layer)
layer=Dropout(0.05)(layer)
layer=Dense(1,activation='sigmoid')(layer)
model=Model(inputs=input,outputs=layer)
model.summary()

# 3. compile,training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,epochs=1
          ,batch_size=len(x),validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=True,restore_best_weights=True))

# 4. predict,evaluate
print(f'accuaracy : {accuracy_score(y_test,np.round(model.predict(x_test)))}')
plt.subplot(1,2,1)
plt.plot(hist.history['val_acc'],label='val_acc')
plt.plot(hist.history['acc'],label='acc')
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.show()