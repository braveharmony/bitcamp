import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

# a=np.arange(0,2*np.pi,0.01)
# b=np.sin(a)
# plt.plot(a,b,color='#e35f62',linestyle="solid")
# plt.grid(True)
# plt.show()

# a=np.array([[[0,2,3]*1 for _ in range(10)]*1 for _ in range(15)])
# print(a.shape)

# a=np.array([1,2])
# b=np.array([a*i for i in range(1,6)])
# print(b)

# a=np.arange(0,5,1)
# b=np.array([a*0.1*i for i in range(100)])
# c=2*np.exp(np.arange(0,3,0.03))
# plt.plot(np.arange(0,1,0.01),b)
# plt.plot(np.arange(0,1,0.01),c)
# plt.show()

# a=np.array([np.arange(0,1,0.1)])
# b=a.reshape(-1,1)
# print((b@a).shape)
# b=a.T
# print((b@a).shape)