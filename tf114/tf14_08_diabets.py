import tensorflow as tf
tf.compat.v1.set_random_seed(1)
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

#1. Data prepare
x,y=load_diabetes(return_X_y=True)
y=y.reshape(-1,1)

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=20580,train_size=0.7)


xp=tf.compat.v1.placeholder(tf.float32)
yp=tf.compat.v1.placeholder(tf.float32)

w1 = tf.compat.v1.Variable(tf.compat.v1.random_normal([10, 16]), name='weight1')
b1 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name='bias1')
layer1 = tf.compat.v1.matmul(xp, w1) + b1                                                           

layer1 = tf.compat.v1.nn.leaky_relu(layer1, alpha=0.5)

w2 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 16]), name='weight2')
b2 = tf.compat.v1.Variable(tf.compat.v1.zeros([16]), name='bias2')
layer2 = tf.compat.v1.matmul(layer1, w2) + b2

layer2 = tf.compat.v1.nn.leaky_relu(layer2, alpha=0.5)

w3 = tf.compat.v1.Variable(tf.compat.v1.random_normal([16, 1]), name='weight2')
b3 = tf.compat.v1.Variable(tf.compat.v1.zeros([1]), name='bias2')
layer3 = tf.compat.v1.matmul(layer2, w3) + b3

hypothesis = layer3

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis-yp))
# optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.1)
train=optimizer.minimize(loss)

min_loss = np.inf
counter = 0
patience = 500  # 원하는 patience 값을 설정하세요.
weights = {
    'w1': None,
    'b1': None,
    'w2': None,
    'b2': None,
    'w3': None,
    'b3': None
}

epochs=2000
sess.run(tf.compat.v1.global_variables_initializer())
for step in range(1,epochs+1):
    sess.run(train,feed_dict={xp:x_train,yp:y_train})
    current_loss = sess.run(loss, feed_dict={xp: x_test, yp: y_test})
    if step%100==0:
        print(f'step:{step}\nw,b:{sess.run([w1,b1])}\nloss:{sess.run(loss,feed_dict={xp:x_test,yp:y_test})}')

    if current_loss < min_loss:
        min_loss = current_loss
        counter = 0
        weights['w1'], weights['b1'], weights['w2'], weights['b2'], weights['w3'], weights['b3'] = sess.run([w1, b1, w2, b2, w3, b3])
    else:
        counter += 1

    if counter >= patience:
        print("Early stopping")
        break

# 최적 가중치로 복원
sess.run(w1.assign(weights['w1']))
sess.run(b1.assign(weights['b1']))
sess.run(w2.assign(weights['w2']))
sess.run(b2.assign(weights['b2']))
sess.run(w3.assign(weights['w3']))
sess.run(b3.assign(weights['b3']))


from sklearn.metrics import r2_score,mean_squared_error
print(f'결정계수 : {r2_score(y_test,sess.run(hypothesis,feed_dict={xp:x_test}))}\nmse : {mean_squared_error(y_test,sess.run(hypothesis,feed_dict={xp:x_test}))}')