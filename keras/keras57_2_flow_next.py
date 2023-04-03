import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()
train_generator=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augment_size=100

x_data=train_generator.flow(np.tile(x_train[0].reshape(28*28),augment_size).reshape(-1,28,28,1)
                            ,np.zeros(augment_size)
                            ,batch_size=augment_size
                            ,shuffle=True).next()
print(x_data)
print(x_data[0].shape)
print(x_data[1].shape)


import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(1,50):
    plt.subplot(7,7,i)
    plt.axis('off')
    # plt.imshow(x_data[0][0][i],cmap='gray')
    plt.imshow(x_data[0][i],cmap='gray')
plt.show()