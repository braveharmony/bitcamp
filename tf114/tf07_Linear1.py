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
x_train=[1,2,3]
y_train=[1,2,3]


# 2. model build
w=tf.compat.v1.Variable(111,dtype=tf.float32)
b=tf.compat.v1.Variable(0,dtype=tf.float32)

x = tf.compat.v1.placeholder(tf.float32)
y = tf.compat.v1.placeholder(tf.float32)


# y=xw+b
hypothesis = x * w + b

# 3-1. complie
mse=tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
mae=tf.compat.v1.reduce_mean(tf.abs(hypothesis-y))

# optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)


train = optimizer.minimize(loss=mse)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


epochs = 100000
for epoch in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_train,y:y_train})
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {sess.run(mse,feed_dict={x:x_train,y:y_train})}, W: {sess.run(w)}, B: {sess.run(b)}")

x_test=[4,5,6]
y_test=[4,5,6]

y_pred = sess.run(hypothesis, feed_dict={x: x_test, y: y_test})
sess.close()
from sklearn.metrics import mean_squared_error
print(f'mse : {mean_squared_error(y_pred,y_test)}')