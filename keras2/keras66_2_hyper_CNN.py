import tensorflow as tf
import numpy as np
import pandas as pd
import random
import os
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input,Dropout
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from tensorflow.keras.optimizers import Adam,Adadelta,RMSprop
# 0.seed initialization
seed=42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['PYTHONHASHSEED']=str(seed)

# 1.data prepare
(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train=x_train.reshape(*x_train.shape,1).astype('float32')/255.
x_test=x_test.reshape(*x_test.shape,1).astype('float32')/255.



# 2. model build
def build_model(drop=0.5,optimizer='adam',activation='relu'
               ,filters=[64,64,64,64],nodes=[64,64,64],lr=0.001
               ,input_shape=x_train.shape[1:]
               ,output_shape=len(np.unique(y_train)))->Sequential:
    inputs=Input(shape=input_shape,name='input')
    x=inputs
    x=Conv2D(filters[0],kernel_size=(3,3),padding='same',activation=activation)(x)
    x=MaxPool2D()(x)
    x=Conv2D(filters[1],kernel_size=(3,3),padding='same',activation=activation)(x)
    x=MaxPool2D()(x)
    x=Conv2D(filters[2],kernel_size=(2,2),padding='valid',activation=activation)(x)
    x=MaxPool2D()(x)
    x=Conv2D(filters[3],kernel_size=(3,3),padding='valid',activation=activation)(x)
    x=Flatten()(x)
    for i,v in enumerate(nodes):
        x=Dense(v,activation=activation,name=f'hiddens{i}')(x)
        x=Dropout(drop)(x)
    x=Dense(256,activation=activation,name='hidden2')(x)
    outputs=Dense(output_shape,activation='softmax',name='outputs')(x)
    model=Model(inputs=inputs,outputs=outputs)
    if type(optimizer)!=str:
        optimizer=optimizer(learning_rate=lr)
    model.compile(optimizer=optimizer,metrics=['acc',]
                  ,loss='sparse_categorical_crossentropy')
    return model

def create_hyperparameter():
    batchs=[100,200,300,400,500]
    optimizers=['adam','rmsprop','adadelta']
    # optimizers=[Adam,Adadelta,RMSprop]
    dropouts=[0.2,0.3,0.4,0.5]
    activations=['relu','elu','selu','linear','swish']
    learning_rates=[0.001,0.0001,0.00001]
    return{'batch_size':batchs,
           'optimizer':optimizers,
           'drop':dropouts,
           'activation':activations,
           'filters':([128,128,128,128]),
           'nodes':([64,64,64],[128,64,128,64]),
           'lr':learning_rates}
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model = KerasClassifier(build_fn=build_model, verbose=True)

# model=GridSearchCV(model,create_hyperparameter(),cv=3)
model=RandomizedSearchCV(model,create_hyperparameter(),cv=2,n_iter=10,verbose=1)
import time
start_time=time.time()
model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20)
print(f'runtime : {time.time()-start_time}')
print(f'best parameters : {model.best_params_}')
print(f'currens model : {model}')
print(f'best estimater : {model.best_estimator_}')
print(f'best score : {model.best_score_}')
print(f'valdiation score at best model : {model.best_estimator_.score(x_test,y_test)}')
print(f'valdiation score at current model: {model.score(x_test,y_test)}')