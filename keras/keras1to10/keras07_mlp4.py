import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. 데이터
x_train = np.array([range(10), range(21,31), range(201,211)]).T
y_train = np.array([[1,2,3,4,5,6,7,8,9,10]]).T

# 2. 모델 생성
model=Sequential()
model.add(Dense(16,input_dim=3))
model.add(Dense(1))

# 3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=10,epochs=5000)

# 4. 모델 평가
loss=model.evaluate(x_train,y_train,batch_size=10)
print(f'loss : {loss}')
result=model.predict([[9,30,210]])
print(f'[[9,30,210]]의 예측값 : {result}')