import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
tf.random.set_seed(337)
np.random.seed(337)

# 1. data prepare
x_train=[1,2,3,4,5]
y_train=[2,4,6,8,10]


# 2. model build
w=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
b=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
print(sess.run(w))

x=tf.compat.v1.placeholder(tf.float32)
y=tf.compat.v1.placeholder(tf.float32)


hypothesis=x*w+b

mse=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y))

optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)

train=optimizer.minimize(loss=mse)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=10000

for epoch in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_train,y:y_train})
    if epoch%100==0:
        print(f'epoch : {epoch}, loss : {sess.run(mse,feed_dict={x:x_train,y:y_train})}, w : {sess.run(w)}, b : {sess.run(b)}')
