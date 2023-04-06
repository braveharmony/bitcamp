import tensorflow as tf
import numpy as np
import random
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout,LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import RobustScaler

# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. x_train prepare
train_datagen=ImageDataGenerator(
    rescale=1
)
test_datagen=ImageDataGenerator(
    rescale=1/255
)

xy_train=train_datagen.flow_from_directory('d:/study_data/_data/brain/train/'
                                  ,target_size=(100,100)
                                  ,batch_size=5
                                  ,class_mode='categorical'
                                  ,color_mode='grayscale'
                                #   ,color_mode='rgb'
                                  ,shuffle=True
                                  )

xy_test=test_datagen.flow_from_directory('d:/study_data/_data/brain/test/'
                                  ,target_size=(100,100)
                                  ,batch_size=5
                                  ,class_mode='categorical'
                                  ,color_mode='grayscale'
                                  ,shuffle=True
                                  )

x_train=(xy_train[i][0] for i in range(len(xy_train)))
y_train=(xy_train[i][1] for i in range(len(xy_train)))
x_train=np.array(list(x_train))
x_train=np.reshape(x_train,([x_train.shape[0]*x_train.shape[1]]+list(x_train.shape[2:])))
y_train=np.array(list(y_train))
y_train=np.reshape(y_train,([y_train.shape[0]*y_train.shape[1]]+list(y_train.shape[2:])))
x_test=(xy_test[i][0] for i in range(len(xy_test)))
y_test=(xy_test[i][1] for i in range(len(xy_test)))
x_test=np.array(list(x_test))
x_test=np.reshape(x_test,([x_test.shape[0]*x_test.shape[1]]+list(x_test.shape[2:])))
y_test=np.array(list(y_test))
y_test=np.reshape(y_test,([y_test.shape[0]*y_test.shape[1]]+list(y_test.shape[2:])))
print(x_train.shape)
print(y_train.shape)
print(y_train[:10])


augment_size=4000
randindx=np.random.randint(x_train.shape[0],size=augment_size)
print(len(np.unique(randindx)))
print(np.min(randindx),np.max(randindx))

 
x_augmented = np.array(x_train[randindx])
y_augmented = np.array(y_train[randindx])

print(x_train.shape)

train_generator=ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=1.2,
    zoom_range=0.3,
    fill_mode='nearest'
)

x_augmented,y_augmented=train_generator.flow(
    x_augmented,y_augmented,shuffle=True,batch_size=augment_size
    # save_to_dir='path'
).next()



print(x_augmented.shape)
print(f'x_train min,max : {np.min(x_train)},{np.max(x_train)}\nx_augmented min,max : {np.min(x_augmented)},{np.max(x_augmented)}')
x_train=np.concatenate((x_train/255.,x_augmented),axis=0)
x_test=x_test/255.
y_train=np.concatenate((y_train,y_augmented),axis=0)



path='d:/study_data/_save/keras58/_5/'
np.save(f'{path}x_train.npy',x_train)
np.save(f'{path}x_test.npy',x_test)
np.save(f'{path}y_train.npy',y_train)
np.save(f'{path}y_test.npy',y_test)
