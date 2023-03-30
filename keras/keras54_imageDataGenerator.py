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

# 1. data prepare
train_datagen=ImageDataGenerator(
    rescale=1/255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(
    rescale=1/255
)

xy_train=train_datagen.flow_from_directory('d:/study_data/brain/train/'
                                  ,target_size=(200,200)
                                  ,batch_size=5
                                  ,class_mode='binary'
                                  ,color_mode='grayscale'
                                  ,shuffle=True
                                  )

xy_test=test_datagen.flow_from_directory('d:/study_data/brain/test/'
                                  ,target_size=(200,200)
                                  ,batch_size=5
                                  ,class_mode='binary'
                                  ,color_mode='grayscale'
                                  ,shuffle=True
                                  )

gendata=(xy_train[i][0] for i in range(len(xy_train)))
gentarget=(xy_train[i][1] for i in range(len(xy_train)))
data=np.array(list(gendata))
data=np.reshape(data,([data.shape[0]*data.shape[1]]+list(data.shape[2:])))
target=np.array(list(gentarget))
target=np.reshape(target,([target.shape[0]*target.shape[1]]+list(target.shape[2:])))
print(data.shape)
print(target.shape)
print(target[:10])