import matplotlib.pyplot as plt
from tensorflow.python.keras.activations import sigmoid,relu,leaky_relu
import numpy as np
x=np.arange(-6,6,0.1)
y_sig=sigmoid(x)
y_relu=relu(x)
y_leakyrelu=leaky_relu(x,alpha=0.5)
plt.plot(x,y_leakyrelu)
plt.grid()
plt.show()