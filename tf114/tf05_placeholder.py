import os
import tensorflow as tf
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
print(sess.run(add_node,feed_dict={a:[1,2],b:[3,4]}))
import numpy as np
print(sess.run(add_node,feed_dict={a:np.array([1,2]),b:np.array([3,4])}))
import pandas as pd
print(sess.run(add_node,feed_dict={a:pd.Series([1,2]),b:pd.Series([3,4])}))

add_and_triple=add_node*3
print(add_and_triple)
print(sess.run(add_and_triple,feed_dict={a:[1,2],b:[3,4]}))
import numpy as np
print(sess.run(add_and_triple,feed_dict={a:np.array([1,2]),b:np.array([3,4])}))
import pandas as pd
print(sess.run(add_and_triple,feed_dict={a:pd.Series([1,2]),b:pd.Series([3,4])}))
