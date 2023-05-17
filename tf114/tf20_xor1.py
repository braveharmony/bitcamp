import tensorflow as tf
import traceback
import numpy as np
if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()

x1=[0,1]
x2=[0,1]
x_data=[[x1[i],x2[j]]for i in range(2)for j in range(2)]
y_data=[[x1[i]^x2[j]]for i in range(2)for j in range(2)]

x=tf.compat.v1.placeholder(tf.float32,shape=[None,2])
y=tf.compat.v1.placeholder(tf.float32,shape=[None,1])

w1=tf.compat.v1.Variable(tf.compat.v1.random_normal([2,32],dtype=tf.float32))
b1=tf.compat.v1.Variable(tf.compat.v1.random_normal([32],dtype=tf.float32))
layer1=tf.compat.v1.matmul(x,w1)+b1

layer1=tf.compat.v1.nn.relu(layer1)

w2=tf.compat.v1.Variable(tf.compat.v1.random_normal([32,1],dtype=tf.float32))
b2=tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32))
layer2=tf.compat.v1.matmul(layer1,w2)+b2

hypothesis=tf.compat.v1.nn.sigmoid(layer2)

loss=-tf.compat.v1.reduce_mean(y*tf.compat.v1.log(hypothesis)+(1-y)*tf.compat.v1.log(1-hypothesis))
optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(loss)

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=1000
for steps in range(1,epochs+1):
    sess.run(train,feed_dict={x:x_data,y:y_data})
    if steps%100==0:
        print(f'steps : {steps}\nloss : {sess.run(loss,feed_dict={x:x_data,y:y_data})}')
        print(f'val_acc : {sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.cast(hypothesis>0.5,dtype=tf.float32),y),dtype=tf.float32)),feed_dict={x:x_data,y:y_data})}')
        print(f'y_pred : {sess.run(hypothesis,feed_dict={x:x_data})}')
        print(f'y_true : {y_data}')
    

from sklearn.metrics import accuracy_score
print(f'acc : {sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.cast(hypothesis>0.5,dtype=tf.float32),y),dtype=tf.float32)),feed_dict={x:x_data,y:y_data})}')
print(f'acc : {accuracy_score(y_data,np.round(sess.run(hypothesis,feed_dict={x:x_data})))}')
sess.close()