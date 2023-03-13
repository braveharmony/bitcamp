import tensorflow as tf
import numpy as np
import random
import matplotlib as mpl
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from matplotlib import pyplot as plt

# 1. data prepare
datasets = load_breast_cancer()
print(datasets.DESCR)
fea=datasets['feature_names']
x=datasets['data']
y=datasets["target"]
print(x.shape,y.shape)
print(fea)

x_train,x_test,y_train,y_test=train_test_split(x,y,shuffle=True,train_size=0.8,random_state=0)

# 2. model build
model=Sequential()
model.add(Dense(32,input_dim=x.shape[1],activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

# 3. compile, training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc','mae'])
es=EarlyStopping(monitor='val_loss',mode='min',patience=50,restore_best_weights=True)
hist=model.fit(x,y,batch_size=len(x)//3,verbose=True,validation_split=0.2,callbacks=[es],epochs=11)


# # 4. evaluation,prediction
y_predict=model.predict(x_test)
# print("=====================================================================")
# print(y_test[:5])
# print(y_predict[:5])
# print("=====================================================================")
y_predict=np.rint(y_predict)


print(f'accuracy_score : {accuracy_score(y_test,y_predict)}')
print(hist.history.keys())

# for i in range(len(fea)):
#     plt.figure(i+1)
#     plt.scatter(x.T[i].T,range(len(x)))
#     plt.title(fea[i])
# plt.show()