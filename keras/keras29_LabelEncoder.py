# 0. seed
import random
import numpy as np
import tensorflow as tf
seed=4
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# 1. 데이터
import pandas as pd
path="./_data/dacon_wine/"
path_save='./_save/dacon_wine/'
df=pd.read_csv(path+'train.csv',index_col=0)
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'sample_submission.csv')
df=df.dropna()
x=df.drop([df.columns[0]], axis=1)
y=df[df.columns[0]]
y=np.array(pd.get_dummies(y,prefix='number'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder,RobustScaler
# from imblearn.over_sampling import RandomOverSampler
le=LabelEncoder()
le.fit(df[df.columns[-1]])
x[df.columns[-1]]=le.transform(x[df.columns[-1]])
dft[df.columns[-1]]=le.transform(dft[df.columns[-1]])
# ros = RandomOverSampler()
# x, y = ros.fit_resample(x, y)
# print(np.unique(x['type'],return_counts=True))
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,stratify=y)
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 2. 모델 구성
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,LeakyReLU,Dropout
# 2. model build
input1=Input(shape=(x.shape[1],))
layer = Dense(32,activation=LeakyReLU(0.35))(input1)
layer = Dropout(0.1)(layer)
layer = Dense(32,activation=LeakyReLU(0.35))(layer)
layer = Dropout(0.1)(layer)
layer = Dense(32,activation=LeakyReLU(0.35))(layer)
layer = Dropout(0.1)(layer)
layer = Dense(32,activation=LeakyReLU(0.35))(layer)
layer = Dropout(0.1)(layer)
# layer = Dense(32,activation=LeakyReLU(0.25))(layer)
# layer = Dropout(0.1)(layer)
# layer = Dense(32,activation=LeakyReLU(0.25))(layer)
# layer = Dropout(0.1)(layer)
# layer = Dense(32,activation=LeakyReLU(0.85))(layer)
# layer = Dropout(0.1)(layer)
# layer = Dense(32,activation=LeakyReLU(0.85))(layer)
# layer = Dropout(0.1)(layer)
layer = Dense(y.shape[1],activation='softmax')(layer)
model=Model(inputs=input1,outputs=layer)
# model.summary()

# 3. 컴파일,훈련
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
hist=model.fit(x_train,y_train,batch_size=len(x_train),epochs=10000
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=200,restore_best_weights=True,verbose=True))

# # 4. 평가, 예측
from sklearn.metrics import accuracy_score
loss=model.evaluate(x_test,y_test)
print(f'loss : {loss}\naccuarcy : {accuracy_score(np.argmax(y_test,axis=1)+3,np.argmax(model.predict(x_test),axis=1)+3)}')

dfs[df.columns[0]]=np.argmax(model.predict(dft),axis=1)+3
dfs.to_csv('./_save/dacon_wine/03_15/forsub.csv',index=False)

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(hist.history['val_loss'],label='val_loss')
plt.plot(hist.history['loss'],label='loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(hist.history['val_acc'],label='val_acc')
plt.plot(hist.history['acc'],label='acc')
plt.legend()
plt.show()
# for i in range(len(x[0])):
#     # plt.figure(keys[i])
#     plt.subplot(int(len(x[0])//2.5)+1,5,2*i+1)
#     plt.scatter(x.T[i],y,s=1)
#     min_x=min(x.T[i]);max_x=max(x.T[i]);min_y=min(y);max_y=max(y)
#     plt.xlim(min_x-0.05*abs(max_x-min_x),max_x+0.05*abs(max_x-min_x));plt.xlabel(keys[i],fontsize=8)
#     plt.ylim(min_y-0.05*abs(max_y-min_y),max_y+0.05*abs(max_y-min_y));plt.ylabel(keys[9],fontsize=8)
#     plt.title(keys[i],fontsize=8)
# plt.show()
