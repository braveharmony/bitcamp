import os
import os
import tensorflow as tf
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)
if tf.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())

sess=tf.compat.v1.Session()

node1=tf.constant(3.0,tf.float32)
node2=tf.constant(3.0)
node3=tf.add(node1,node2)

a=tf.compat.v1.placeholder(tf.float32)
b=tf.compat.v1.placeholder(tf.float32)

add_node=a+b
print(add_node)
print(sess.run(add_node,feed_dict={a:3,b:4.5}))
print(sess.run(add_node,feed_dict={a:[1,3],b:[2,4]}))
print(sess.run(add_node,feed_dict={a:np.array([1,3]),b:np.array([2,4])}))
print(sess.run(add_node,feed_dict={a:pd.Series([1,3]),b:pd.Series([2,4])}))

add_and_triple=add_node*3
print(add_and_triple)
print(sess.run(add_and_triple,feed_dict={a:7,b:3}))
print(sess.run(add_and_triple,feed_dict={a:[1,3],b:[2,4]}))
print(sess.run(add_and_triple,feed_dict={a:np.array([1,3]),b:np.array([2,4])}))
print(sess.run(add_and_triple,feed_dict={a:pd.Series([1,3]),b:pd.Series([2,4])}))


