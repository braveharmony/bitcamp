import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh,elu
def custom_elu(x):
    return np.where(x > 0, x, (np.exp(x) - 1))
x=np.arange(-5,5,0.1)
plt.subplot(1,2,1)
plt.plot(x,elu(x),label='keras elu')
plt.title('keras elu')
plt.subplot(1,2,2)
plt.plot(x,custom_elu(x),label='custom elu')
plt.title('custom elu')
plt.legend()
plt.show()