import tensorflow as tf
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
tf.random.set_seed(337)
np.random.seed(337)

# 1. data prepare
x=[1,2,3,4,5]
y=[2,4,6,8,10]


# 2. model build
input=tf.compat.v1.placeholder(tf.float32)
input_layer=input

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
weight_layer1=input_layer*w1+b1

relu_layer1=tf.nn.relu(weight_layer1)

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
weight_layer2=relu_layer1*w2+b2

output_layer=weight_layer1
output=tf.compat.v1.placeholder(tf.float32)


# 3. complie,trainging
mse=tf.compat.v1.reduce_mean(tf.compat.v1.square(output_layer-output))
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss=mse)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=10000
for epoch in range(1,epochs+1):
    sess.run(train,feed_dict={input:x,output:y})
    if epoch%100==0:
        print(f'epoch : {epoch}, loss : {sess.run(mse,feed_dict={input:x,output:y})}\n w1 : {sess.run(w1)}, b1 : {sess.run(b1)}\n w2 : {sess.run(w2)}, b2 : {sess.run(b2)}')
