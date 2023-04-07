import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Dense,Dropout,Conv1D,SimpleRNN,Concatenate,LeakyReLU,Flatten,LSTM,MaxPool1D
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

# # 모델 빌드할때 쓸 작은 파일
# folder='Crossing!'
# filenum=1
# startload=time.time()
# audio_signal=np.load(f"d:/study_data/_project/_data/audio_signal/{folder}.npy")
# rfftx=np.load(f"d:/study_data/_project/_data/r_fftx/{folder}r_fftx.npy")
# ifftx=np.load(f"d:/study_data/_project/_data/i_fftx/{folder}i_fftx.npy")
# mfcc=np.load(f"d:/study_data/_project/_data/mfcc/{folder}mfcc.npy")
# sample_rate=np.load(f"d:/study_data/_project/_data/sr/{folder}sr.npy")
# y=np.load(f"d:/study_data/_project/_data/ys/{folder}ys.npy")
# print(f'loadtime : {time.time()-startload}')

# fit 할때 쓸 전체 파일
startload=time.time()
audio_signal=np.load(f"d:/study_data/_project/_data/audio_signal.npy")
print(f'audio signal loadtime : {time.time()-startload}')
startload=time.time()
rfftx=np.load(f"d:/study_data/_project/_data/r_fftx.npy")
print(f'rfftx loadtime : {time.time()-startload}')
startload=time.time()
ifftx=np.load(f"d:/study_data/_project/_data/i_fftx.npy")
print(f'ifftx loadtime : {time.time()-startload}')
startload=time.time()
mfcc=np.load(f"d:/study_data/_project/_data/mfcc.npy")
print(f'mfcc loadtime : {time.time()-startload}')
startload=time.time()
sample_rate=np.load(f"d:/study_data/_project/_data/sr.npy")
y=np.load(f"d:/study_data/_project/_data/ys.npy")
print(f'y loadtime : {time.time()-startload}')


# 1-1.train_test_split
audio_signal_train,audio_signal_test,rfftx_train,rfftx_test,\
ifftx_train,ifftx_test,mfcc_train,mfcc_test,y_train,y_test\
    =train_test_split(audio_signal,rfftx,ifftx,mfcc,y,train_size=0.8,random_state=seed,stratify=y,shuffle=True)
x_train=(audio_signal_train,rfftx_train,ifftx_train,mfcc_train)
x_test=(audio_signal_test,rfftx_test,ifftx_test,mfcc_test)



# 2. model build
Input_data=Input(shape=(audio_signal.shape[1:]),name='input1')
Input_rfftx=Input(shape=(rfftx.shape[1:]),name='input2')
Input_ifftx=Input(shape=(ifftx.shape[1:]),name='input3')
Input_mfcc=Input(shape=(mfcc.shape[1:]),name='input4')

# 2-1 Input_data layer
layer1=Conv1D(128,sample_rate//50,strides=sample_rate//200,name='input1-1',activation=LeakyReLU(0.25))(Input_data)
layer1=Conv1D(128,16,strides=2,name='input1-2',activation=LeakyReLU(0.25))(layer1)
layer1=Conv1D(128,16,strides=2,name='input1-3',activation=LeakyReLU(0.25))(layer1)
layer1=Conv1D(32,64,name='input1-5',activation=LeakyReLU(0.25))(layer1)
layer1=LSTM(128)(layer1)
layer1=Flatten()(layer1)
layer1=Dense(64,activation=LeakyReLU(0.25))(layer1)
layer1=Dropout(0.125)(layer1)
layer1=Dense(32,activation=LeakyReLU(0.25))(layer1)

# 2-2 Input_rfftx layer
layer2=Conv1D(128,sample_rate//50,strides=sample_rate//200,name='input2-1',activation=LeakyReLU(0.25))(Input_rfftx)
layer2=Dropout(0.5)(layer2)
layer2=Conv1D(128,16,strides=2,name='input2-2',activation=LeakyReLU(0.25))(layer2)
layer2=Conv1D(128,16,strides=2,name='input2-3',activation=LeakyReLU(0.25))(layer2)
layer2=Conv1D(128,32,name='input2-5',activation=LeakyReLU(0.25))(layer2)
layer2=Flatten()(layer2)
layer2=Dense(64,activation=LeakyReLU(0.25))(layer2)
layer2=Dense(32,activation=LeakyReLU(0.25))(layer2)

# 2-3 Input_ifftx layer
layer3=Conv1D(128,sample_rate//50,strides=sample_rate//200,name='input3-1',activation=LeakyReLU(0.25))(Input_ifftx)
layer3=Dropout(0.5)(layer3)
layer3=Conv1D(128,16,strides=2,name='input3-2',activation=LeakyReLU(0.25))(layer3)
layer3=Conv1D(128,16,strides=2,name='input3-3',activation=LeakyReLU(0.25))(layer3)
layer3=Conv1D(128,32,name='input3-5',activation=LeakyReLU(0.25))(layer3)
layer3=Flatten()(layer3)
layer3=Dense(64,activation=LeakyReLU(0.25))(layer3)
layer3=Dense(32,activation=LeakyReLU(0.25))(layer3)

# 2-4 Input_mfcc layer
layer4=Conv1D(128,32,name='input4-1',activation=LeakyReLU(0.25))(Input_mfcc)
layer4=Conv1D(128,8,strides=2,name='input4-2',activation=LeakyReLU(0.25))(layer4)
layer4=Conv1D(128,8,name='input4-3',activation=LeakyReLU(0.25))(layer4)
layer4=Conv1D(64,32,strides=2,name='input4-5',activation=LeakyReLU(0.25))(layer4)
layer4=LSTM(128)(layer4)
layer4=Dense(64,activation=LeakyReLU(0.25))(layer4)
layer4=Dropout(0.125)(layer4)
layer4=Dense(32,activation=LeakyReLU(0.25))(layer4)


# 2-5 Concatenate layer
layer5=Concatenate()((layer1,layer2,layer3,layer4))
layer5=Dense(64,activation=LeakyReLU(0.25))(layer5)
layer5=Dropout(0.125)(layer5)
layer5=Dense(64,activation=LeakyReLU(0.25))(layer5)
layer5=Dropout(0.125)(layer5)
layer5=Dense(64,activation=LeakyReLU(0.25))(layer5)
output=Dense(y.shape[1],activation='softmax')(layer5)

# 2-6 input,output connect and summary
model=Model(inputs=(Input_data,Input_rfftx,Input_ifftx,Input_mfcc),outputs=output)
model.summary()

# 3. compile,training
from tensorflow.keras.callbacks import EarlyStopping
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
hist=model.fit(x_train,y_train,batch_size=4,epochs=10000
          ,validation_data=(x_test,y_test),callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=50,verbose=True,restore_best_weights=True))
val_acc=hist.history['val_acc'][-1]
model.save(f'./__project/_model_save/model_acc_{val_acc}.h5')