import autokeras as ak
from keras.datasets import cifar10
from typing import Tuple
import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

datasets:Tuple[Tuple[np.ndarray],...]=cifar10.load_data()
(x_train,y_train),(x_test,y_test)=datasets

model=ak.ImageClassifier(max_trials=2)

model.fit(x_train,y_train,epochs=10,batch_size=100,validation_data=(x_test,y_test))