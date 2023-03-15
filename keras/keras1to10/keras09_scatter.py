import numpy as np
import tensorflow as tf
import random

# 0.seed
# seed=2
# np.random.seed(seed)
# random.seed(seed)
# tf.random.set_seed(seed)

# 1. Data

x = np.array([i for i in range(1,21)])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y
                                ,train_size=0.6,shuffle=True,random_state=1234)

# 2. Model Build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
model=Sequential()
model.add(Dense(8,input_dim=1,activation='linear'))
model.add(Dense(16,activation='sigmoid'))
model.add(Dense(16,activation=LeakyReLU()))
model.add(Dense(16,activation='linear'))
model.add(Dense(4,activation='linear'))
model.add(Dense(1))

# 3. Compile, training
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=100,epochs=1000)

# 4. evaluation,predict
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')

xx=np.array([np.linspace(min(x)-3,max(x)+3,100)]).T
yy_predict=model.predict(xx)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.scatter(x_train,y_train,s=10,c='red',label='train data')
plt.scatter(x_test,y_test,s=10,c='green',label='test data')
plt.plot(xx,model.predict(xx),label='model shape')
plt.legend()
plt.show()