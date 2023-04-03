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
datagen=ImageDataGenerator(
    rescale=1/255,
)

path='d:/study_data/_data/horse-or-human'
target_size=(100,100)
xy=datagen.flow_from_directory(directory=path
                                  ,target_size=target_size
                                  ,batch_size=1027
                                  ,class_mode='categorical'
                                #   ,color_mode='grayscale'
                                  ,color_mode='rgb'
                                  ,shuffle=True
                                  )


# x=(xy[i][0] for i in range(len(xy)))
# y=(xy[i][1] for i in range(len(xy)))
# x=np.array(list(x))
# print(x.shape)
# x=np.reshape(x,([x.shape[0]*x.shape[1]]+list(x.shape[2:])))
# y=np.array(list(y))
# y=np.reshape(y,([y.shape[0]*y.shape[1]]+list(y.shape[2:])))

x=xy[0][0]
y=xy[0][1]
print(x.shape)
print(y.shape)
print(x[:5])
print(y[:5])

print(f'runtime for generate : {time.time()-save_start}')

save_start=time.time()
path='d:/study_data/_save/horse-or-human/'
np.save(file=f'{path}x.npy',arr=x)
np.save(file=f'{path}y.npy',arr=y)


print(f'runtime for save : {time.time()-save_start}')

save_start=time.time()
path='d:/study_data/_save/horse-or-human/'
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)
