import tensorflow as tf
tf.compat.v1.set_random_seed(337)
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
x_data=[[73,51,65],
        [92,98,11],
        [89,31,33],
        [99,33,100],
        [17,66,79]]
y_data=list([i] for i in [152,185,180,205,142])

x=tf.compat.v1.placeholder(tf.float32)
y=tf.compat.v1.placeholder(tf.float32)

w=tf.compat.v1.Variable(tf.compat.v1.random_normal([]),name='weight')
b=tf.compat.v1.Variable(tf.compat.v1.random_normal([]),name='bias')

hypotheis= x * w + b
sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

print(sess.run(hypotheis,feed_dict={x:x_data,y:y_data}))