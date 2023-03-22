import tensorflow as tf
import pandas as pd
import matplotlib as mpl
import numpy as np
import random,time,datetime
from tensorflow.python.keras.models import Sequential


dataset = range(1,11)
timesteps = 5

def split_x(dataset,timesteps):
    forlistgen=(dataset[i : i+timesteps] for i in range(len(dataset)- timesteps + 1))#제너레이터 메소드
    return np.array(list(forlistgen))
    return np.array([dataset[i : i+timesteps] for i in range(len(dataset)- timesteps + 1)])#리스트 컴프리헨션

x=split_x(dataset,timesteps)
print(x.shape)