import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU
from tensorflow.keras.callbacks import Callback
tf.random.set_seed(337)
#1. 데이터
x=np.array(range(5))+1
y=x.copy()


class PrintWeights(Callback):
    def __init__(self,**para):
        super(PrintWeights, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        print(f'layer 0 : {self.model.layers[0].get_weights()}')
        print(f'layer 1 : {self.model.layers[1].get_weights()}')

#2. 
model=Sequential()
model.add(Dense(2,input_dim=1))
model.add(Dense(1))

layer0=model.layers[0].get_weights()
layer1=model.layers[1].get_weights()

for i in x:
    y=np.matmul(i*layer0[0]+layer0[1],layer1[0])+layer1[1]
    print('===========================================')
    print(f'y=({i}*{layer0[0]}+{layer0[1]})*{layer1[0]}+{layer1[1]}')
    print(f'y={y[0]}')