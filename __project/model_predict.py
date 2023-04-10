import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,SimpleRNN,Concatenate,LeakyReLU,Flatten,LSTM,MaxPool2D
from sklearn.model_selection import train_test_split
import time,datetime
from tensorflow.keras.utils import Sequence
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except RuntimeError as e:
        print(e)

# gpus=tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(
#             gpus[0],
#             [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]
#         )
#     except RuntimeError as e:
#         print(e)
            

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare

def file_path(file_extension, filenum, folder=''):
    return f"d:/study_data/_project/_data/{folder}/{filenum}.{file_extension}"

# 모델 테스트할때 쓸 파일
model=load_model('./__project/_model_save/stft_model_test.h5')

music_sample=('Crossing!','Dreaming!','Flyers!!!','GlowMap','Harmony4You','Rainbow','UNION!!')
index = list(set(range(7))-set(np.load(f"d:/study_data/_project/_data/musicindex.npy")))
musics=(music_sample[i] for i in index)
idolnum = np.load(file_path("npy", "idolnum"))
count=0
correct=0
for folder in musics:
    for filenum in idolnum:
        sample_rate = np.load(file_path("npy", f"{filenum}sr", folder))
        stft=np.load(file_path("npy", f"{filenum}stft", folder))
        y_true = np.load(file_path("npy", f"{filenum}ys", folder))

        # 2. model build
        y_pred=model.predict(stft)
        y_pred=np.sum(y_pred,axis=0)
        y_pred=np.argmax(y_pred,axis=0)
        y_true=np.sum(y_true,axis=0)
        y_true=np.argmax(y_true,axis=0)
        count+=1
        if y_pred==y_true:
            correct+=1
print(f'acc : {correct/count}')