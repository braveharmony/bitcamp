import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()


sess=tf.compat.v1.Session()
x=tf.Variable([2],dtype=tf.float32)
y=tf.Variable([3],dtype=tf.float32)


init=tf.compat.v1.global_variables_initializer()
sess.run(init)


print(sess.run(x+y))