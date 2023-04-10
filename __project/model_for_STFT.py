import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import Input,Dense,Dropout,Conv2D,SimpleRNN,Concatenate,LeakyReLU,Flatten,LSTM,MaxPool2D
from sklearn.model_selection import train_test_split
import time
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

# # 모델 빌드할때 쓸 작은 파일
# folder = 'Crossing!'
# startload = time.time()
# sample_rate = np.load(file_path("npy", f"{folder}sr", "sr"))
# stft=np.load(file_path("npy", f"{folder}stft", "stft"))
# y = np.load(file_path("npy", f"{folder}ys", "ys"))
# print(np.unique(np.argmax(y,axis=1)))
# print(f'loadtime : {time.time()-startload}')

# fit 할때 쓸 전체 파일
startload = time.time()
stft = np.load(file_path("npy", "stft"))
print(f'mfcc loadtime : {time.time()-startload}')

startload = time.time()
sample_rate = np.load(file_path("npy", "sr"))
y = np.load(file_path("npy", "ys"))
print(f'y loadtime : {time.time()-startload}')

# 1-1.train_test_split
x_train,x_test,y_train,y_test=train_test_split(stft,y,train_size=0.9,random_state=seed,stratify=y,shuffle=True)


# 2. model build
model=Sequential()
model.add(Input(shape=x_train.shape[1:]))
model.add(Conv2D(128,(8,4),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D())
model.add(Conv2D(256,(4,4),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D())
model.add(Conv2D(512,(8,4),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D())
model.add(Conv2D(256,(3,1),padding='valid',activation=LeakyReLU(0.25)))
model.add(MaxPool2D())
model.add(Conv2D(512,(2,2),padding='same',activation=LeakyReLU(0.25)))
model.add(MaxPool2D())
model.add(Conv2D(256,(2,2),padding='valid',activation=LeakyReLU(0.25)))
model.add(Flatten())
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(256,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(256,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(128,activation=LeakyReLU(0.25)))
model.add(Dropout(1/16))
model.add(Dense(52,activation='softmax'))

model.summary()

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
hist=model.fit(x_train,y_train,batch_size=32,epochs=10000,shuffle=True
          ,validation_data=(x_test,y_test),callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=True,restore_best_weights=True))
acc=model.evaluate(x_test,y_test)[1]


import datetime
date=datetime.datetime.now()
date = date.strftime('%m월%d일 %H시%M분')
model.save('./__project/_model_save/stft_model_test.h5')

del x_train,x_test

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

import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(range(len(hist.history['val_loss'])),hist.history['val_loss'],label='val_loss')
plt.plot(range(len(hist.history['loss'])),hist.history['loss'],label='loss')
plt.ylim((min(hist.history['loss']),max(hist.history['loss'])))
plt.legend()
plt.subplot(1,2,2)
plt.plot(range(len(hist.history['val_acc'])),hist.history['val_acc'],label='val_acc')
plt.plot(range(len(hist.history['acc'])),hist.history['acc'],label='acc')
plt.legend()
plt.show()