import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_seed(123)
if tf.compat.v1.executing_eagerly:
    tf.compat.v1.disable_eager_execution()
x=[1,2,3]
y=[1,2,3]
input_tensor=tf.compat.v1.placeholder(tf.float32)
output_tensor=tf.compat.v1.placeholder(tf.float32)

w=tf.compat.v1.Variable([10],dtype=tf.float32)

hypothesis=input_tensor*w
loss=tf.compat.v1.reduce_mean(tf.square(hypothesis-output_tensor))

lr=0.1
gredient=tf.compat.v1.reduce_mean(2*(hypothesis-output_tensor)*x)

descent= w-lr*gredient
update=w.assign(descent)

w_history=[]
loss_history=[]

sess=tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epochs=20
print(f'w[0]:{sess.run(w).__iter__().__next__()}')
for step in range(1,epochs+1):
    _,cul_loss,cul_w=sess.run([update,loss,w],feed_dict={input_tensor:x,output_tensor:y})
    print(f'{step}\t{cul_loss}\t{cul_w}')
    w_history.append(cul_w)
    loss_history.append(cul_loss)
    
x_test=[4,5,6]
y_test=[4,5,6]
y_predict=sess.run(hypothesis,feed_dict={input_tensor:x_test})
from sklearn.metrics import r2_score,mean_squared_error as mse
print(f'결정계수 : {r2_score(y_test,y_predict)}\nMSE : {mse(y_test,y_predict)}')
sess.close()

# print(f'w_hist:{w_history}')
# print(f'loss_hist:{loss_history}')
# plt.subplot(1,2,1)
# plt.plot(range(1,epochs+1),w_history)
# plt.title('w_history')
# plt.subplot(1,2,2)
# plt.plot(range(1,epochs+1),loss_history)
# plt.title('loss_history')
# plt.show()