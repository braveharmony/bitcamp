import tensorflow as tf
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
# 1. data prepare
import pandas as pd
datasets=pd.read_csv('./_data/dacon_diabete/train.csv',index_col=0)
x_data=datasets.drop(datasets.columns[-1],axis=1)
y_data=datasets[datasets.columns[-1]].values.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_data,x_test,y_data,y_test=train_test_split(x_data,y_data,train_size=0.8)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_data=scaler.fit_transform(x_data)
x_test=scaler.transform(x_test)

# 2. model build
x=tf.compat.v1.placeholder(tf.float32,shape=[None,8])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal([8,16]),dtype=tf.float32)
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal([16]),dtype=tf.float32)
layer1=tf.compat.v1.matmul(x,w1)+b1

layer1=0.5*tf.compat.v1.nn.relu(layer1)+0.5

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal([16,1]),dtype=tf.float32)
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
layer2=tf.compat.v1.matmul(layer1,w2)+b2

hypothesis=tf.compat.v1.sigmoid(layer2)

epsilon=1e-15
# loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y))
loss = -tf.compat.v1.reduce_mean(y*tf.compat.v1.math.log(hypothesis+epsilon)+(1-y)*tf.compat.v1.math.log(1-hypothesis+epsilon))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001)

train=optimizer.minimize(loss=loss)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs=5000
for step in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    if step%100==0:
        print(f'step:{step},loss:{sess.run(loss,feed_dict={x:x_test,y:y_test})}')

from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np
print(f'결정계수:{r2_score(y_test,sess.run(hypothesis,feed_dict={x:x_test}))},acc:{accuracy_score(y_test,sess.run(tf.compat.v1.cast(hypothesis>0.5,tf.float32),feed_dict={x:x_test}))}')