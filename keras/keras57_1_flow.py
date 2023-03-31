from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train,y_train),(x_test,y_test)=fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
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

print(x_train.shape)
print(np.tile(x_train[0],augment_size).reshape(-1,x_train.shape[1],x_train.shape[2],1).shape)
x_data=train_datagen(np.tile(x_train[0],augment_size).reshape(-1,x_train.shape[1],x_train.shape[2],1).shape,
        np.zeros(augment_size))
print(np.array(x_data).shape)
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
    plt.subplot(7,7,i+1)
    plt.axis('off')
    plt.imshow(x_data[0][i],cmap='gray')