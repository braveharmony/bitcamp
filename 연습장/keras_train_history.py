# 0. seed initialization
import numpy as np
import tensorflow as tf
import random 
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

datasets = fetch_california_housing()
x=datasets.data
y=datasets.target

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
model=Sequential(Dense(64,input_shape=x.shape[1:]))
model=Sequential(Dense(1))


# 3. compile,training
from tensorflow.python.keras.callbacks import EarlyStopping
model.compile(loss='mse',optimizer='adam')

hist=model.fit(x_train,y_train,epochs=1000,batch_size=len(x_train)//10
          ,validation_split=0.2,verbose=True
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min'
                                   ,restore_best_weights=True,patience=5))

# 4. predict,evaluate
from sklearn.metrics import r2_score
print(f'결정계수 : {r2_score(y_test,model.predict(x_test))}')

# 5. history ploting
from matplotlib import pyplot as plt
plt.plot(hist.history['loss'],label='loss')
plt.plot(hist.history['val_loss'],label='val_loss')
plt.legend()
plt.show()