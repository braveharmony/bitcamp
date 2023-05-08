import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
graph = tf.Graph()

with graph.as_default():
    node1=tf.compat.v1.constant(3.0,tf.float32)
    node2=tf.compat.v1.constant(4.0)

    sess=tf.compat.v1.Session(graph=graph)
    print(node1)
    print(node2)
    print(node1+node2)
    print(sess.run(node1+node2))