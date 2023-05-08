import os
import tensorflow as tf

if tf.__version__[0]=='2':
    tf.compat.v1.disable_eager_execution()
    
    
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

node1=tf.constant(3.0,tf.float32)
node2=tf.constant(4.0)

sess=tf.compat.v1.Session()
print(node1)
print(node2)
print(node1+node2)
print(sess.run(node1+node2))