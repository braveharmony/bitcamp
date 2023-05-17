import tensorflow as tf
import numpy as np
import matplotlib as mpl
import pandas as pd
import random, time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
# 1. data prepare
path='./_data/DDarung/'
df=pd.read_csv(path+'train.csv',index_col=0)
df=df.dropna()
# print(df.describe())
# print(df.isnull().sum())
dft=pd.read_csv(path+'test.csv',index_col=0)
dfs=pd.read_csv(path+'submission.csv')
# print(df.columns,dft.columns)
x=df.drop([df.columns[-1]],axis=1)
y=df[df.columns[-1]].values.reshape(-1,1)
# print(np.unique(y))
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=seed,shuffle=True)
scaler=MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
dft=scaler.transform(dft)

# 2. model build
x=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,x_train.shape[1]])
y=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,y_train.shape[1]])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[x_train.shape[1],32],dtype=tf.float32))
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32],dtype=tf.float32))
layer1=tf.compat.v1.nn.relu(tf.compat.v1.matmul(x,w1)+b1)

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32,64],dtype=tf.float32))
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[64],dtype=tf.float32))
layer2=tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer1,w2)+b2)

w3=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[64,32],dtype=tf.float32))
b3=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32],dtype=tf.float32))
layer3=tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer2,w3)+b3)

w4=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32,y_train.shape[1]],dtype=tf.float32))
b4=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[y_train.shape[1]],dtype=tf.float32))
hypothesis=tf.compat.v1.matmul(layer3,w4)+b4

# 3. compile training
sess=tf.compat.v1.Session()
loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(loss)
sess.run(tf.compat.v1.global_variables_initializer())


epochs=32

start_time=time.time()
for steps in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_train,y:y_train})
    print(f'steps : {steps} loss : {sess.run(loss,feed_dict={x:x_train,y:y_train})}')
    
# 4. evaluate predict
from sklearn.metrics import r2_score
print(f'loss : {sess.run(loss,feed_dict={x:x_test,y:y_test})}')
print(f'결정계수 : {r2_score(y_test,sess.run(hypothesis,feed_dict={x:x_test}))}')
print(f'runtime : {time.time()-start_time}')
sess.close()



