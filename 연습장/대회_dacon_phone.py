import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout,Lambda
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path='./_data/dacon_phone/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
# print(df.info())
# print(dft.info())
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]
print(np.unique(y,return_counts=True)[1])

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 1. model build
model=Sequential()
model.add(Dense(64,activation=LeakyReLU(0.5),input_shape=(x.shape[1],)))
model.add(Dropout(0.125))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(1,activation='sigmoid'))
model.summary()


# 2. compile, build
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_train,y_train,verbose=True
          ,epochs=10000
          ,validation_split=0.2,batch_size=len(x)//3
          ,class_weight={0:2*3318/(13441+3318),1:26882/(13441+3318)}
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=300,restore_best_weights=True,verbose=True))

# 3. predict,evaluate
y_predict=np.round(model.predict(x_test))
print(f'accuracy:{accuracy_score(y_test,y_predict)}\nF1_score : {f1_score(y_test,y_predict)}')

plt.subplot(1,2,1)
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['acc'],label='acc')
plt.plot(hist.history['val_acc'],label='val_acc')
plt.legend()
plt.show()