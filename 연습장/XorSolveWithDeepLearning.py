import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib as mpl
from matplotlib import pyplot as plt
x=np.array([[0,0],[-1,0],[0,-1],[-1,-1],[0,1],[-1,1],[0,2],[-1,2],[1,0],[1,-1],[2,0],[2,-1],[1,1],[1,2],[2,1],[2,2]])
y=np.array([0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0])

model=Sequential()
model.add(Dense(16,input_dim=2,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='mae',optimizer='adam')
model.fit(x,y,epochs=5000)

# 4. 결과 파일 평가
loss=model.evaluate(x,y)
print(f'loss : {loss}')
result=model.predict(x)
resultstr=str()

# 5. 결과 출력
for i in range(len(result)):
    resultstr+=f'x:{x[i]},y:{y[i]},result:{result[i]}\n'
print(resultstr)

# 6. 플로팅
plt.scatter(x[:,0],x[:,1],c=result)
plt.show()