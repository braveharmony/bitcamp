import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print(tf.__version__)

print('Hello World!')


if tf.__version__[0]=='2':
    tf.compat.v1.disable_eager_execution()
    
tensor=tf.compat.v1.constant('Hello World!')
print(tensor)

sess=tf.compat.v1.Session()
print(sess.run(tensor).decode('utf-8'))

# graph = tf.Graph()
# with graph.as_default():
#     tensor=tf.compat.v1.constant('Hello World!')
# print(tensor)

# print(tf.compat.v1.Session(graph=graph).run(tensor).decode('utf-8'))
# sess=tf.compat.v1.Session(graph=graph)
# print(sess.run(tensor).decode('utf-8'))