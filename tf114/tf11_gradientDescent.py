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
    print(f'w[{step}]:w[{step-1}]-lr*2*(hypothesis-y)*x={sess.run(w-lr*gredient,feed_dict={input_tensor:x,output_tensor:y}).__iter__().__next__()}')
    _,cul_loss,cul_w=sess.run([update,loss,w],feed_dict={input_tensor:x,output_tensor:y})
    # print(f'{step}\t{cul_loss}\t{cul_w}')
    w_history.append(cul_w)
    loss_history.append(cul_loss)
sess.close()
'''
w[0]:10.0
w[1]:w[0]-lr*2*(hypothesis-y)*x=1.6
w[2]:w[1]-lr*2*(hypothesis-y)*x=1.04
w[3]:w[2]-lr*2*(hypothesis-y)*x=1.002666711807251
w[4]:w[3]-lr*2*(hypothesis-y)*x=1.0001777410507202
w[5]:w[4]-lr*2*(hypothesis-y)*x=1.000011920928955
w[6]:w[5]-lr*2*(hypothesis-y)*x=1.0000008344650269
w[7]:w[6]-lr*2*(hypothesis-y)*x=1.0000001192092896
w[8]:w[7]-lr*2*(hypothesis-y)*x=1.0
w[9]:w[8]-lr*2*(hypothesis-y)*x=1.0
w[10]:w[9]-lr*2*(hypothesis-y)*x=1.0
w[11]:w[10]-lr*2*(hypothesis-y)*x=1.0
w[12]:w[11]-lr*2*(hypothesis-y)*x=1.0
w[13]:w[12]-lr*2*(hypothesis-y)*x=1.0
w[14]:w[13]-lr*2*(hypothesis-y)*x=1.0
w[15]:w[14]-lr*2*(hypothesis-y)*x=1.0
w[16]:w[15]-lr*2*(hypothesis-y)*x=1.0
w[17]:w[16]-lr*2*(hypothesis-y)*x=1.0
w[18]:w[17]-lr*2*(hypothesis-y)*x=1.0
w[19]:w[18]-lr*2*(hypothesis-y)*x=1.0
w[20]:w[19]-lr*2*(hypothesis-y)*x=1.0
'''
print(f'w_hist:{w_history}')
print(f'loss_hist:{loss_history}')
plt.subplot(1,2,1)
plt.plot(range(1,epochs+1),w_history)
plt.title('w_history')
plt.subplot(1,2,2)
plt.plot(range(1,epochs+1),loss_history)
plt.title('loss_history')
plt.show()