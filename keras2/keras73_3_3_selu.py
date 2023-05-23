import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh,selu
def custom_selu(x):
    return np.where(x > 0, x, (np.exp(x) - 1))
x=np.arange(-5,5,0.1)
plt.subplot(1,2,1)
plt.plot(x,selu(x),label='keras selu')
plt.title('keras selu')
plt.subplot(1,2,2)
plt.plot(x,custom_selu(x),label='custom selu')
plt.title('custom selu')
plt.legend()
plt.show()
print(selu(x)/custom_selu(x))