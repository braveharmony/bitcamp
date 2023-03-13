import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
from matplotlib import pyplot as plt

# 0. seed initialization
seed=4
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data
path='./_data/dacon_diabete/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
# print(df.columns)
# print(dft.columns)
# print(dfs.columns)
# print(df.isnull().sum())
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed)
x_train=x
y_train=y
# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dense(16,activation=LeakyReLU(0.5)))
model.add(Dense(32,activation=LeakyReLU(0.5)))
model.add(Dense(1,activation="sigmoid"))
# model.summary()


# 3. compile,train
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True)
hist=model.fit(x_train,y_train,batch_size=len(x_train),verbose=True,validation_split=0.2
               ,callbacks=es
               ,epochs=5000)

# 4. evaluate,save
y_predict=np.round(model.predict(x_test))
print(f'accuracy : {accuracy_score(y_test,y_predict)}')

# 5. save
y_predict=np.round(model.predict(dft))
dfs[df.columns[-1]]=y_predict
# print(dfs)
pathsave='./_save/dacon_diabete/03_10/'
dfs.to_csv(pathsave+'forsub.csv',index=False)

plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'])
plt.title('binary_crossentropy')
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'])
plt.title('val_acc')
plt.show()
