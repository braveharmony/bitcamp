import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
tf.random.set_seed(123)
if tf.compat.v1.executing_eagerly:
    tf.compat.v1.disable_eager_execution()

x=[1,2,3]
y=[1,2,3]
w=tf.compat.v1.placeholder(tf.float32)
input_tensor=tf.compat.v1.placeholder(tf.float32)

output_tensor=w*input_tensor

loss=tf.compat.v1.reduce_mean(tf.compat.v1.square(output_tensor-y))

w_hist=[]
loss_hist=[]

with tf.compat.v1.Session() as sess:
    sess:tf.compat.v1.Session
    for i in range(-30,50):
        curr_w=i
        curr_loss=sess.run(loss,feed_dict={input_tensor:x,w:curr_w})
        w_hist.append(curr_w)
        loss_hist.append(curr_loss)
print(f'w_hist:{w_hist}')
print(f'loss_hist:{loss_hist}')
plt.subplot(1,2,1)
plt.plot(range(-30,50),w_hist)
plt.title('w_history')
plt.subplot(1,2,2)
plt.plot(range(-30,50),loss_hist)
plt.title('loss_history')
plt.show()

# ws=(-30,-10,0,10,30,50)
# x=[1,2]
# y=[1,2]
# losses=[]
# for w in ws:
#     loss=np.mean(np.square(np.array(x)*w-np.array(y)))
#     losses.append(loss)
#     print(f'w:{w},loss:{loss}')
# plt.scatter(ws,losses)
# plt.show()