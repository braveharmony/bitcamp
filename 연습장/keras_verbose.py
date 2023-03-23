# 1. data prepare
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
datasets=fetch_california_housing()
x=datasets.data
y=datasets.target
print(f'x.shape : {x.shape} y.shape : {y.shape}')

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.9,random_state=0)
print(f'x_train.shape : {x_train.shape} x_test.shape :{x_test.shape}')
print(f'y_train.shape : {y_train.shape} y_test.shape : {y_test.shape}')

# 2. model build
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,LeakyReLU
model=Sequential()
model.add(Dense(16,input_dim=8,activation='linear'))
model.add(Dense(1))
# 3. Compile, training
model.compile(loss='mse',optimizer='adam')

model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=-1)
print('===============================================verbose=-1\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=-0.1)
print('===============================================verbose=-0.1\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=0)
print('===============================================verbose=0\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=0.5)
print('===============================================verbose=0.5\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=1)
print('===============================================verbose=1\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=2)
print('===============================================verbose=2\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=3)
print('===============================================verbose=3\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=True)
print('===============================================verbose=True\n')
model.fit(x_train,y_train,batch_size=100,epochs=1,verbose=False)
print('===============================================verbose=False\n')