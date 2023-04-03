import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from matplotlib import pyplot as plt

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

plt.imshow(x_train[1])
plt.title(y_train[1])
plt.show()