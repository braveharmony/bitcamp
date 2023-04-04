import tensorflow as tf
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,RobustScaler
# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data prepare
path = 'd:/study_data/_data/air_pressure/'
df=pd.read_csv(path+'train_data.csv')
print(df)
x=df.drop([df.columns[2],df.columns[-1]],axis=1)
y=df[df.columns[-1]]
encoder=OneHotEncoder()
print()
y=np.array([y]).T
y=encoder.fit_transform(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y,shuffle=True,random_state=seed)

scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

# 2. model selectiom
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,LeakyReLU,Input
model=Sequential()
model.add(Input(shape=(x_train.shape[1:])))
model.add(Dense(128,))