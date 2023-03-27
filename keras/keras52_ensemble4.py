import tensorflow as tf
import random
import numpy as np

# 0. seed initialziation
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
gen_x1=([i,i+301]for i in range(100))
# gen_x2=([i+101,i+411,i+150]for i in range(100))
# gen_x3=([i+201,i+511,i+1300]for i in range(100))
x1=np.array(list(gen_x1))
# x2=np.array(list(gen_x2))
# x3=np.array(list(gen_x3))
print(x1.shape)


y1 = np.array(range(2001,2101))
y2 = np.array(range(1001,1101))


x_train,x_test,y1_train,y1_test,y2_train,y2_test=train_test_split(x1,y1,y2,train_size=0.7,shuffle=True,random_state=seed)

# def spliter123(x,x1,x2,x3):
#     return x[:,:x1.shape[1]],x[:,x1.shape[1]:x1.shape[1]+x2.shape[1]],x[:,x1.shape[1]+x2.shape[1]:]
# x1_train,x2_train,x3_train=spliter123(x_train,x1,x2,x3)
# x1_test,x2_test,x3_test=spliter123(x_test,x1,x2,x3)

scaler=MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
 
# x1_train=scaler.fit_transform(x1_train)
# x1_test=scaler.transform(x1_test)
# x2_train=scaler.fit_transform(x2_train)
# x2_test=scaler.transform(x2_test)
# x3_train=scaler.fit_transform(x3_train)
# x3_test=scaler.transform(x3_test)


# 2. model1 build
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Concatenate
# 2-1 input1
input1=Input(shape=(x1.shape[1]))
layer1=Dense(64,activation='linear')(input1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)

# 2-2 input2
# input2=Input(shape=(x2.shape[1]))
# layer2=Dense(64,activation='linear')(input2)
# layer2=Dense(64,activation='linear')(layer2)
# layer2=Dense(64,activation='linear')(layer2)
# layer2=Dense(64,activation='linear')(layer2)
# # 2-3 input3
# input3=Input(shape=(x3.shape[1]))
# layer3=Dense(64,activation='linear')(input3)
# layer3=Dense(64,activation='linear')(layer3)
# layer3=Dense(64,activation='linear')(layer3)
# layer3=Dense(64,activation='linear')(layer3)
# from tensorflow.python.keras.layers import concatenate,Concatenate
# 2-3 concatenate
# merge1=concatenate(inputs=(layer1,layer2,layer3))
output1=Dense(64)(layer1)
output1=Dense(64)(output1)
output1=Dense(64)(output1)
output1=Dense(1)(output1)
output2=Dense(64)(layer1)
output2=Dense(64)(output2)
output2=Dense(64)(output2)
output2=Dense(1)(output2)
model1=Model(inputs=(input1,),outputs=(output1,output2))
model1.summary()

# 2.model2. build
model2=Sequential()
model2.add(Dense(64,input_shape=(x1.shape[1],)))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(64))
model2.add(Dense(2))

# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
import time
model1.compile(loss='mse',optimizer='adam')
start1=time.time()
model1.fit(x_train,(y1_train,y2_train)
          ,epochs=1000,batch_size=len(x_train)//40
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',verbose=True
                                   ,patience=50,restore_best_weights=True))
runtime1=time.time()-start1
model2.compile(loss='mse',optimizer='adam')
start2=time.time()
model2.fit(x_train,np.concatenate(([y1_train],[y2_train]),axis=0).T
          ,epochs=1000,batch_size=len(x_train)//40
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',verbose=True
                                   ,patience=50,restore_best_weights=True)
            )
runtime2=time.time()-start2

# 4. predict,evaluate
from sklearn.metrics import r2_score,mean_squared_error
def RMSE(y_true,y_pred):
    gen=(np.sqrt(mean_squared_error(y_true[i],y_pred[i]))for i in range(len(y_true)))   
    return np.mean(list(gen))
def r2(y_true,y_pred):
    gen=(r2_score(y_true[i],y_pred[i])for i in range(len(y_true)))
    return np.mean(list(gen))

print('======================model1======================')
y_test=(y1_test,y2_test)
y_pred1=np.array(model1.predict(x_test))
print(f'runtime : {runtime1}\n결정계수 : {r2(y_test,y_pred1)}\nRMSE : {RMSE(y_test,y_pred1)}\n')

print('======================model2======================')

y_pred2=np.array(model2.predict(x_test)).T
print(f'runtime : {runtime2}\n결정계수 : {r2(y_test,y_pred2)}\nRMSE : {RMSE(y_test,y_pred2)}\n')
