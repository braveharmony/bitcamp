import tensorflow as tf
tf.random.set_seed(337)

if tf.compat.v1.executing_eagerly():
    tf.compat.v1.disable_eager_execution()
x=tf.compat.v1.placeholder(tf.float32,shape=[None])
y=tf.compat.v1.placeholder(tf.float32,shape=[None])
w=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
b=tf.compat.v1.Variable(tf.compat.v1.random_normal([1]),dtype=tf.float32)
x_data=tf.compat.v1.placeholder(tf.float32,shape=[None])

hypothesis=x*w+b

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())


loss=tf.compat.v1.reduce_mean(tf.square(hypothesis-y))
optimizer=tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(loss)


epochs=1000
for step in range(1,epochs+1):
    _,loss_val,w_val,b_val=sess.run([train,loss,w,b]
                                    ,feed_dict={x:[1,2,3,4,5]
                                                ,y:[2,4,6,8,10]})
    if step %20 == 0:
            print(step, loss_val, w_val, b_val)
x_test=[6,7,8]  
y_test=[12,14,16]
y_pred=sess.run(hypothesis,feed_dict={x:x_test})
from sklearn.metrics import r2_score
print(f'y_pred: {y_pred} 결정계수 : {r2_score(y_pred,y_test)}')