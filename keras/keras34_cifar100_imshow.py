import tensorflow as tf
from tensorflow.keras.datasets import cifar100
from matplotlib import pyplot as plt

# 1. data prepare
(x_train,y_train),(x_test,y_test)=cifar100.load_data()

for i in range(500):
    if i%100==0:
        plt.figure(1+i//100)
    plt.subplot(10,10,i%100+1)
    plt.imshow(x_train[i])
    plt.title(y_train[i])
plt.show()