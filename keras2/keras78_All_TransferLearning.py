import tensorflow as tf
import numpy as np
import pandas as pd
import time,os,random
from tensorflow.keras.applications import VGG16,VGG19
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications import ResNet101,ResNet101V2,ResNet152,ResNet152V2
from tensorflow.keras.applications import DenseNet201,DenseNet121,DenseNet169
from tensorflow.keras.applications import InceptionV3,InceptionResNetV2
from tensorflow.keras.applications import MobileNet,MobileNetV2
from tensorflow.keras.applications import MobileNetV3Small,MobileNetV3Large
from tensorflow.keras.applications import NASNetMobile,NASNetLarge
from tensorflow.keras.applications import EfficientNetB0,EfficientNetB1,EfficientNetB7
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
model_list = [VGG16, VGG19, ResNet50, ResNet50V2, ResNet101, ResNet101V2, ResNet152, ResNet152V2, DenseNet201, DenseNet121, DenseNet169, InceptionV3, InceptionResNetV2, MobileNet, MobileNetV2, MobileNetV3Small, MobileNetV3Large, NASNetMobile, NASNetLarge, EfficientNetB0, EfficientNetB1, EfficientNetB7, Xception
]

# 0. seed initialization 
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

for model_class in model_list:
    model:Model=model_class()
    model.trainable=False
    print('==============================')
    print(f'모델명 : {model_class.__name__}')
    print(f'전체 가중치 갯수 : {sum([np.prod(i.shape) for i in model.weights])}')
    print(f'훈련 가능 갯수 : {sum([np.prod(i.shape) for i in model.trainable_weights])}')