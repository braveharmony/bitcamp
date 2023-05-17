import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random, time
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris
from matplotlib import pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

# 1. data prepare
datasets=load_iris()
x=datasets.data
y=datasets.target

# print(x.shape,y.shape)
# print(np.unique(y))

y=pd.get_dummies(y,prefix='number')

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,stratify=y,random_state=seed)
MMS=MinMaxScaler().fit(x_train)
x_train=MMS.transform(x_train)
x_test=MMS.transform(x_test)

# 2. model build
x=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,x_train.shape[1]])
y=tf.compat.v1.placeholder(dtype=tf.float32,shape=[None,y_train.shape[1]])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[x_train.shape[1],32],dtype=tf.float32))
b1=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[32],dtype=tf.float32))
layer1=tf.compat.v1.nn.relu(tf.compat.v1.matmul(x,w1)+b1)

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32,64],dtype=tf.float32))
b2=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[64],dtype=tf.float32))
layer2=tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer1,w2)+b2)

w3=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[64,32],dtype=tf.float32))
b3=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[32],dtype=tf.float32))
layer3=tf.compat.v1.nn.relu(tf.compat.v1.matmul(layer2,w3)+b3)

w4=tf.compat.v1.Variable(tf.compat.v1.random_normal(shape=[32,y_train.shape[1]],dtype=tf.float32))
b4=tf.compat.v1.Variable(tf.compat.v1.zeros(shape=[y_train.shape[1]],dtype=tf.float32))
hypothesis=tf.compat.v1.nn.sigmoid(tf.compat.v1.matmul(layer3,w4)+b4)

# 3. compile, training
sess=tf.compat.v1.Session()
loss=-tf.compat.v1.reduce_mean(tf.compat.v1.reduce_sum(y*tf.compat.v1.log(hypothesis+1e-7),axis=1))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(loss)
sess.run(tf.compat.v1.global_variables_initializer())

start_time=time.time()
epochs=32
for steps in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_train,y:y_train})
    print(f'steps : {steps} loss : {sess.run(loss,feed_dict={x:x_train,y:y_train})}')
    
runtime=time.time()-start_time
# 4. predict
y_predict=sess.run(hypothesis,feed_dict={x:x_test,y:y_test})
y_test=np.argmax(y_test.values,axis=1)
y_predict=np.argmax(y_predict,axis=1)
print(f'accuracy : {accuracy_score(y_test,y_predict)}')
print(f'runtime : {runtime}')
