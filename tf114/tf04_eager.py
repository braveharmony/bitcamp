import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)
if tf.__version__[0]=='2':
    tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly())   #False

