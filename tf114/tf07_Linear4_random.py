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
