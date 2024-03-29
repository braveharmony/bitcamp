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
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.7,
    zoom_range=0.1,
    fill_mode='nearest'
)

augment_size=40000
randindx=np.random.randint(x_train.shape[0],size=augment_size)
print(len(np.unique(randindx)))
print(np.min(randindx),np.max(randindx))

x_train=np.reshape(x_train,list(x_train.shape)+[1])
x_test=np.reshape(x_test,list(x_test.shape)+[1])
 
x_augmented = np.array(x_train[randindx])
y_augmented = np.array(y_train[randindx])

print(x_train.shape)

x_augmented,y_augmented=train_generator.flow(
    x_augmented,y_augmented,shuffle=True,batch_size=augment_size
).next()



print(x_augmented.shape)
print(f'x_train min,max : {np.min(x_train)},{np.max(x_train)}\nx_augmented min,max : {np.min(x_augmented)},{np.max(x_augmented)}')
x_train=np.concatenate((x_train/255.,x_augmented),axis=0)
x_test=x_test/255.
# x_train=np.array(list(x_train/255)+list(x_augmented))
y_train=np.concatenate((y_train,y_augmented),axis=0)
print(x_train.shape)