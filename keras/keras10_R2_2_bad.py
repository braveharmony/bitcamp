# 1. R2를 음수가 아닌 0.5 이하로 만들것
# 2. 데이터는 건들지 말것
# 3. 레이어는 인풋 아웃풋 포함 7개 이상
# 4. batch_size=1
# 5. 히든레이어 노드는 10개 이상 100개 이하
# 6. train_size는 75 프로 rhwjd
# 7. epoch 100번 이상
# 8. loss지표는 mse,mae
# [실습]
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib import pyplot as plt

# 0. 시드
seed=0
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# 1. Data
x= np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y= np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.75,random_state=0)

# 2. model build
model=Sequential()
model.add(Dense(8,input_dim=1,))
model.add(Dense(1))

# 3. compile, train
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1,epochs=20)

# 4. evaluation,predict
loss = model.evaluate(x_test,y_test)
print(f'loss : {loss}')
y_predict=model.predict(x_test)
r2=r2_score(y_test,y_predict)
print(f'r2_result : {r2}')