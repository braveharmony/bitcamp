import numpy as np
import matplotlib.pyplot as plt
f = lambda x : x*(x-2)*(x-4)*(x-8)

x=np.linspace(-1,9,100)
y=f(x)

lr=0.001
epochs=2000
gradient=lambda x : x*(x-2)*(x-4)+x*(x-4)*(x-8)+x*(x-2)*(x-8)+(x-2)*(x-4)*(x-8)

opt=lambda w : w-lr*gradient(w)
w=[]
w0=-1.0
w.append(w0)
for i in range(epochs):
    w.append(opt(w[-1]))
w=np.array(w)
loss=f(w)
print(w,loss)
plt.plot(x,y)
plt.scatter(w,loss,c='orange')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()