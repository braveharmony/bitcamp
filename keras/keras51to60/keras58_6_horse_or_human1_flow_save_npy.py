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
    rescale=1,
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

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True,random_state=seed,stratify=y)

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
    rotation_range=5,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.7,
    zoom_range=0.1,
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



path='d:/study_data/_save/keras58/_6/'
np.save(f'{path}x_train.npy',x_train)
np.save(f'{path}x_test.npy',x_test)
np.save(f'{path}y_train.npy',y_train)
np.save(f'{path}y_test.npy',y_test)
print(f'runtime for save : {time.time()-save_start}')
