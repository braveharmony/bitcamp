from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import r2_score


# 0. 시드
# random.seed(seed)
# np.random.seed(seed)
# tf.random.set_seed(seed)

# 1. 데이터 생성
datasets=load_diabetes()
x=datasets.data
y=datasets.target
seed=20580 
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)
print(x.shape,y.shape)

#2. 모델 생성
model=Sequential()
model.add(Dense(1,input_dim=10))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=1000,epochs=6000)

#4.평가
y_predict=model.predict(x_test,batch_size=1000)
r2=r2_score(y_test,y_predict)
print(f'r2 : {r2}')
# seed: 72, r2 : 0.6307189572199093
# seed: 4432, r2 : 0.6516497740480287
# seed: 14392 ,r2: 0.6644751088539174
# seed: 19315 ,r2: 0.6570518881215789
# seed: 20580 ,r2: 0.6767530959498996
# seed: 34553 ,r2: 0.6467952464547478
# seed: 43519 ,r2: 0.6252434470357784
# seed: 49578 ,r2: 0.6377535010466053
# seed: 55498 ,r2: 0.66127024868337
# seed: 72896 ,r2: 0.6609120993720627
# seed: 83172 ,r2: 0.6389428202573828
# seed: 84525 ,r2: 0.6496028460248617
# seed: 87538 ,r2: 0.6427956319504499
# seed: 88463 ,r2: 0.6598242468090456
# seed: 124177 ,r2: 0.6512106586952531
# seed: 134896 ,r2: 0.6452878036818283
# seed: 153793 ,r2: 0.6592285689491271
# seed: 169450 ,r2: 0.6445891001316243
# seed: 180289 ,r2: 0.6416429650170346
# seed: 197722 ,r2: 0.6569129627390211
