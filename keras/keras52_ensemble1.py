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
from sklearn.preprocessing import RobustScaler
gen_x1=([i,i+301]for i in range(100))
gen_x2=([i+101,i+411,i+150]for i in range(100))
x1=np.array(list(gen_x1))
x2=np.array(list(gen_x2))
print(x1.shape,x2.shape)


y = np.array(range(2001,2101))

x1_train,x1_test,x2_train,x2_test,y_train,y_test=train_test_split(x1,x2,y,train_size=0.7,shuffle=True,random_state=seed)
scaler=RobustScaler()
x1_train=scaler.fit_transform(x1_train)
x1_test=scaler.transform(x1_test)
x2_train=scaler.fit_transform(x2_train)
x2_test=scaler.transform(x2_test)



# 2. model1 build
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Concatenate
# 2-1 model1
input1=Input(shape=(x1.shape[1]))
layer1=Dense(64,activation='linear')(input1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)
layer1=Dense(64,activation='linear')(layer1)
# 2-2 model1
input2=Input(shape=(x2.shape[1]))
layer2=Dense(64,activation='linear')(input2)
layer2=Dense(64,activation='linear')(layer2)
layer2=Dense(64,activation='linear')(layer2)
layer2=Dense(64,activation='linear')(layer2)
from tensorflow.python.keras.layers import concatenate
# 2-3 concatenate
merge1=concatenate(inputs=(layer1,layer2))
merge1=Dense(50)(merge1)
output=Dense(1)(merge1)
model1=Model(inputs=(input1,input2),outputs=output)
model1.summary()

# 2.model2. build
model2=Sequential()
model2.add(Dense(64,input_shape=(x1.shape[1]+x2.shape[1],)))
model2.add(Dense(1))

# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
import time
model1.compile(loss='mse',optimizer='adam')
start=time.time()
model1.fit((x1_train,x2_train),y
          ,epochs=5000,batch_size=len(x1_train)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',verbose=True
                                   ,patience=30,restore_best_weights=True))
model2.compile(loss='mse',optimizer='adam')
model2.fit(np.concatenate((x1_train,x2_train),axis=1),y
          ,epochs=5000,batch_size=len(x1_train)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',verbose=True
                                   ,patience=30,restore_best_weights=True)
            )


# 4. predict,evaluate
from sklearn.metrics import r2_score,mean_squared_error
def RMSE(y_true,y_pred):
    return np.sqrt(mean_squared_error(y_true,y_pred))
print('======================model1======================')
y_pred=model1.predict((x1_test,x2_test))
print(f'runtime : {time.time()-start}\n결정계수 : {r2_score(y_test,y_pred)}\nRMSE : {RMSE(y_test,y_pred)}')
print('======================model2======================')
y_pred=model2.predict(np.concatenate((x1_test,x2_test),axis=1))
print(f'runtime : {time.time()-start}\n결정계수 : {r2_score(y_test,y_pred)}\nRMSE : {RMSE(y_test,y_pred)}')