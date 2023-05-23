import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid,tanh,relu
custom_Leakyrelu=lambda x : 0.95*np.maximum(0,x)+0.05*x
x=np.arange(-5,5,0.1)
plt.plot(x,custom_Leakyrelu(x),label='custom Leakyrelu')
plt.title('custom Leakyrelu')
plt.legend()
plt.show()