import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)

if tf.__version__[0]=='2':
    tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   #False



if tf.__version__[0]=='2':
    graph = tf.Graph()
    with graph.as_default():
        print(tf.executing_eagerly())   #False
        print(tf.compat.v1.executing_eagerly())   #False