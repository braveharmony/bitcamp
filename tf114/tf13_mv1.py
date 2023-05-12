import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_seed(123)
if tf.compat.v1.executing_eagerly:
    tf.compat.v1.disable_eager_execution()
x1_data=[73,93,89,96,73]
x2_data=[80,88,91,98,66]
x3_data=[75,93,90,100,70]
y_data=[152,185,180,196,142]

x_data=np.array([x1_data,x2_data,x3_data]).T
y_data=np.array(y_data).reshape(-1, 1)

input=tf.compat.v1.placeholder(tf.float32)
output=tf.compat.v1.placeholder(tf.float32)
w=tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 1]))
b=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]))
hypothesis=tf.compat.v1.matmul(input,w)+b

loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-output))
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.000001)
train=optimizer.minimize(loss)

epochs=30000
sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(1,epochs+1):
    sess.run(train,feed_dict={input:x_data,output:y_data})
    if step%100==0:
        print(f'step:{step}\nw.b:{sess.run([w,b])}\nloss:{sess.run(loss,feed_dict={input:x_data,output:y_data})}')
print(sess.run(hypothesis,feed_dict={input:x_data}))