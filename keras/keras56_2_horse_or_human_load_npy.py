# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import time


# 0. seed initialization
seed = 0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. x prepare

save_start=time.time()
path='d:/study_data/_save/horse-or-human/'
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)

save_start=time.time()
path='d:/study_data/_save/horse-or-human/'
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)
