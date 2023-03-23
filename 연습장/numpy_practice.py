import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
x = np.linspace(-10, 10, 100)
y = sigmoid(x)
plt.subplot(1,2,1)
plt.plot(x, y)
x =np.array(list((i for i in range(-9,10) if i != 0)))
y= np.array(list((1 if i > 0 else 0 for i in x)))
y_hat=sigmoid(x)
plt.scatter(x,y)
plt.scatter(x,y_hat)
plt.title('sigmoid')

def binary_cross_entropy(y, y_hat):
    epsilon = 1e-7  # 0으로 나누는 오류 방지
    bce =  -y * np.log(y_hat + epsilon) - (1 - y) * np.log(1 - y_hat + epsilon)
    return bce
bcd=binary_cross_entropy(y,y_hat)
plt.subplot(1,2,2)
plt.scatter(x,bcd)
plt.title('binary_cross_entropy')
plt.show()