import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input,Dense,LeakyReLU,Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_california_housing