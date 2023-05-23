import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import sigmoid
custom_sigmoid=lambda x : 1/(1+np.exp(-x))
x=np.arange(-5,5,0.1)
# plt.subplot(1,2,1)
plt.plot(x,sigmoid(x),label='keras sigmoid')
# plt.title('keras sigmoid')
# plt.subplot(1,2,2)
plt.plot(x,custom_sigmoid(x),label='custom sigmoid')
# plt.title('custom sigmoid')
plt.legend()
plt.show()