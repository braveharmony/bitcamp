# x has 3 feature and y has 2 feature
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Data
x_train = np.array([range(10), range(21,31), range(201,211)]).T
y_train = np.array([[1,2,3,4,5,6,7,8,9,10],
                    [1+0.1*i for i in range(10)]]).T

# 2. model build
model=Sequential()
model.add(Dense(4,input_dim=3,activation='linear'))
model.add(Dense(4))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))

# 3. compile, fitting
model.compile(loss='mse',optimizer='adam')
model.fit(x_train,y_train,batch_size=10,epochs=3000)

# 4. model evaluation
loss = model.evaluate(x_train,y_train)
print(f'loss : {loss}')
result = model.predict([[9,30,210]])
print(f'[[9,30,210]]의 예상값 : {result}')

