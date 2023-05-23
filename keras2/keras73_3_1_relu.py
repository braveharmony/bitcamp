import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh,relu
custom_relu=lambda x : np.maximum(0,x)
x=np.arange(-5,5,0.1)
plt.subplot(1,2,1)
plt.plot(x,relu(x),label='keras relu')
plt.title('keras relu')
plt.subplot(1,2,2)
plt.plot(x,custom_relu(x),label='custom relu')
plt.title('custom relu')
plt.legend()
plt.show()