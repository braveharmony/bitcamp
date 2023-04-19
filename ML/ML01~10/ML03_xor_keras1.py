import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import random 
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import accuracy_score

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
x1=[0,1]
x2=[0,1]
x_data=[[i,j]for i in x1 for j in x2]
y_data=[i ^ j for i in x1 for j in x2]

# 2. model
model=Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(1))

# 3. compile,training
model.compile(loss='mae',optimizer=SGD(learning_rate=0.01))
model.fit(x_data,y_data,epochs=1000)
print(f'{accuracy_score(y_data,np.round(model.predict(x_data)))}')