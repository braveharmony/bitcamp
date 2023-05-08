import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

print('Hello World!')

tensor=tf.constant('Hello World!')
print(tensor)

print(tf.compat.v1.Session().run(tensor).decode('utf-8'))
sess=tf.Session()
print(sess.run(tensor).decode('utf-8'))