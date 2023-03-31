# https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import time


from PIL import UnidentifiedImageError
import io

# Custom load_img function with exception handling
def custom_load_img(path, grayscale=False, color_mode='rgb', target_size=None,
                    interpolation='nearest'):
    try:
        return load_img(path, grayscale=grayscale, color_mode=color_mode,
                        target_size=target_size, interpolation=interpolation)
    except UnidentifiedImageError:
        print(f"Cannot identify image file {path}, skipping...")
        return None

ImageDataGenerator.load_img = custom_load_img

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

path='d:/study_data/_data/cat_dog/PetImages'

xy=datagen.flow_from_directory(directory=path
                                  ,target_size=(100,100)
                                  ,batch_size=24998
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
path='d:/study_data/_save/cat_dog/'
np.save(file=f'{path}x.npy',arr=x)
np.save(file=f'{path}y.npy',arr=y)


print(f'runtime for save : {time.time()-save_start}')

save_start=time.time()
path='d:/study_data/_save/cat_dog/'
x=np.load(file=f'{path}x.npy')
y=np.load(file=f'{path}y.npy')


print(f'runtime for load : {time.time()-save_start}')
print(x.shape)
print(y.shape)
