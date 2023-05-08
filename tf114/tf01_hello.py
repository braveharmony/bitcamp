import os
import tensorflow as tf
if tf.__version__[0]=='2':
    tf.compat.v1.disable_eager_execution()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

print('Hello World!')

tensor=tf.constant('Hello World!')
print(tensor)

sess=tf.compat.v1.Session()
print(sess.run(tensor).decode('utf-8'))