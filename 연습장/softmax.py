import matplotlib.pyplot as plt
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

x=np.linspace(-10,10,1000)
y=softmax(x)
print(sum(y))
plt.plot(x,y)
plt.show()