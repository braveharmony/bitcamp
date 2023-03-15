import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import random,time,datetime
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Conv2D,Flatten
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

model=Sequential()
model.add(Dense(10,input_shape=(3,)))
model.add(Dense(units=15))
model.summary()

x=Dense(20,input_shape=(3,))
x.units=-3
print(x.units)
print(x._batch_input_shape)
x._batch_input_shape=(16,)
print(x._batch_input_shape)

# Dense 의 constructor는 이렇게 정의되어 있다.
# def __init__(
#         self,
#         units,
#         activation=None,
#         use_bias=True,
#         kernel_initializer="glorot_uniform",
#         bias_initializer="zeros",
#         kernel_regularizer=None,
#         bias_regularizer=None,
#         activity_regularizer=None,
#         kernel_constraint=None,
#         bias_constraint=None,
#         **kwargs,
#     ):
#         super().__init__(activity_regularizer=activity_regularizer, **kwargs)

#         self.units = int(units) if not isinstance(units, int) else units
#         if self.units < 0:
#             raise ValueError(
#                 "Received an invalid value for `units`, expected "
#                 f"a positive integer. Received: units={units}"
#             )
#         self.activation = activations.get(activation)
#         self.use_bias = use_bias
#         self.kernel_initializer = initializers.get(kernel_initializer)
#         self.bias_initializer = initializers.get(bias_initializer)
#         self.kernel_regularizer = regularizers.get(kernel_regularizer)
#         self.bias_regularizer = regularizers.get(bias_regularizer)
#         self.kernel_constraint = constraints.get(kernel_constraint)
#         self.bias_constraint = constraints.get(bias_constraint)

#         self.input_spec = InputSpec(min_ndim=2)
#         self.supports_masking = True