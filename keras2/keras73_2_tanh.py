import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh
hyperbolic_tangent=lambda x : (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
x=np.arange(-5,5,0.1)
plt.subplot(1,3,1)
plt.plot(x,tanh(x),label='keras tanh')
plt.title('keras tanh')
plt.subplot(1,3,2)
plt.plot(x,np.tanh(x),label='numpy tanh')
plt.title('numpy tanh')
plt.subplot(1,3,3)
plt.plot(x,hyperbolic_tangent(x),label='custom hyperbolic tangent')
plt.title('custom hyperbolic tangent')
plt.legend()
plt.show()