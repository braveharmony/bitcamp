import tensorflow as tf
tf.compat.v1.set_random_seed(1)
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

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([3, 5]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([5]), name='bias1')
layer1 = tf.compat.v1.matmul(x, w1) + b1

layer1 = tf.compat.v1.nn.relu(layer1)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([5, 5]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([5]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

layer2 = tf.compat.v1.nn.relu(layer2)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([5, 1]), name='weight2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias2')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

hypothesis = layer3

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-y))
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
train=optimizer.minimize(loss)

epochs=3000
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    if step%100==0:
        print(f'step:{step}\nw,b:{sess.run([w1,b1])}\nloss:{sess.run(loss,feed_dict={x:x_data,y:y_data})}')

from sklearn.metrics import r2_score,mean_squared_error
print(f'결정계수 : {r2_score(y_data,sess.run(hypothesis,feed_dict={x:x_data}))}\nmse : {mean_squared_error(y_data,sess.run(hypothesis,feed_dict={x:x_data}))}')