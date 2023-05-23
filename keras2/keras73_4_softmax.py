import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.activations import softmax
custom_softmax= lambda x : np.exp(x)/np.sum(np.exp(x))
x=np.arange(-5,5,0.5)
# plt.subplot(1,2,1)
# plt.plot(x,softmax(x),label='keras softmax')
# plt.title('keras softmax')
plt.subplot(1,2,1)
plt.pie(custom_softmax(x),custom_softmax(x),shadow='False')
plt.subplot(1,2,2)
plt.plot(x,custom_softmax(x),label='custom softmax')
plt.title('custom softmax')
plt.legend()
plt.show()