import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

train_generator=ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1
    ,shear_range=0.7
    ,zoom_range=0.1,
    fill_mode='nearest',
    vertical_flip=True,
    horizontal_flip=True,
)

augment_size=40000

randindx=range(40000)

x_train=np.reshape(x_train,list(x_train.shape)+[1])
x_test=np.reshape(x_test,list(x_test.shape)+[1])

x_argments=np.array(x_train[randindx])
y_argments=np.array(y_train[randindx])

x_argments,y_argments=train_generator.flow(x_argments,y_argments,batch_size=augment_size,shuffle=False).next()

plt.figure(figsize=(4,5))
for i in range(1,11):
    plt.subplot(4,5,i)
    plt.axis('off')
    plt.imshow(x_train[i],cmap='gray')
    plt.subplot(4,5,10+i)
    plt.axis('off')
    plt.imshow(x_argments[i],cmap='gray')
plt.show()