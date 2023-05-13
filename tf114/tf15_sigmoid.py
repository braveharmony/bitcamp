import tensorflow as tf
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
# 1. data prepare
x_data=[[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data=[[0],[0],[0],[1],[1],[1]]

# 2. model build
x=tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal([2,16]),dtype=tf.float32)
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal([16]),dtype=tf.float32)
layer1=tf.compat.v1.matmul(x,w1)+b1

layer1=tf.compat.v1.nn.relu(layer1)

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal([16,1]),dtype=tf.float32)
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
layer2=tf.compat.v1.matmul(layer1,w2)+b2

hypothesis=tf.compat.v1.sigmoid(layer2)

# loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y))
loss = -tf.compat.v1.reduce_mean(y*tf.compat.v1.math.log(hypothesis)+(1-y)*tf.compat.v1.math.log(1-hypothesis))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)

train=optimizer.minimize(loss=loss)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
epochs=3000
for step in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    if step%100==0:
        print(f'step:{step},loss:{sess.run(loss,feed_dict={x:x_data,y:y_data})}')

from sklearn.metrics import r2_score,accuracy_score,f1_score
import numpy as np
print(f'결정계수:{r2_score(y_data,sess.run(hypothesis,feed_dict={x:x_data}))},acc:{accuracy_score(y_data,sess.run(tf.compat.v1.cast(hypothesis>0.5,tf.float32),feed_dict={x:x_data}))}')