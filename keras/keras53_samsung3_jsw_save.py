# 삼성전자와 현대자동차 주가로 삼성전자 주가 맞추기
# 각각 데이터에서 컬럼 7개 이상 추출(그 중 거래량은 반드시 들어갈 것)
# timesteps와 feature는 알아서 잘라라

# 제공된 데이터 외 추가 데이터 사용 금지

# 1. 삼성전자 28일(화) 종가 맞추기(점수배점 0.3)
# 2. 삼성전자 29일(수) 아침 시가 맞추기(점수배점 0.7)
# 메일 제목 : 장승원 [현대 2차] 60,350.07원
# 첨부 파일 : keras53_samsung2_jsw_submit.py
# 첨부 파일 : keras53_samsung4_jsw_submit.py
# 가중치    : _save/samsung/keras53_samsung2_jsw.h5
# 가중치    : _save/samsung/keras53_samsung4_jsw.h5
import tensorflow as tf
import pandas as pd
import random
import numpy as np
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import LSTM,Dense,Input,Concatenate
import matplotlib.pyplot as plt

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf. random.set_seed(seed)

# 1. data prepare
samsung=pd.read_csv('./_data/시험/삼성전자 주가3.csv', encoding='cp949',index_col=0)
hyundai=pd.read_csv('./_data/시험/현대자동차2.csv', encoding='cp949',index_col=0)
samsung=samsung.drop(samsung.columns[4],axis=1)
hyundai=hyundai.drop(hyundai.columns[4],axis=1)

for col in samsung.columns:
    if samsung[col].dtype == 'object':
        samsung[col] = pd.to_numeric(samsung[col].str.replace(',', ''), errors='coerce')
for col in hyundai.columns:
    if hyundai[col].dtype == 'object':
        hyundai[col] = pd.to_numeric(hyundai[col].str.replace(',', ''), errors='coerce')
samsung=samsung.iloc[:200][::-1]
hyundai=hyundai.iloc[:200][::-1]

# '시가', '고가', '저가', '종가', 'Unnamed: 6', '등락률', '거래량', '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'
col_drop=[7,8,14]

samsung=samsung.drop([samsung.columns[i] for i in col_drop],axis=1)
hyundai=hyundai.drop([hyundai.columns[i] for i in col_drop],axis=1)
print(samsung.info())
print(hyundai.info())


# for i in range(len(samsung.columns)):
#     plt.figure(f'{i}.{samsung.columns[i]}')

#     samsung_col = samsung[samsung.columns[i]]
#     hyundai_col = hyundai[hyundai.columns[i]]

#     samsung_min = samsung_col.min()
#     samsung_max = samsung_col.max()
#     samsung_diff = samsung_max - samsung_min
#     samsung_padding = 0.1 * samsung_diff

#     hyundai_min = hyundai_col.min()
#     hyundai_max = hyundai_col.max()
#     hyundai_diff = hyundai_max - hyundai_min
#     hyundai_padding = 0.1 * hyundai_diff

#     plt.subplot(1, 2, 1)
#     plt.plot(range(len(samsung_col)), np.array(samsung_col[0:len(samsung_col)]))
#     plt.yticks(np.arange(samsung_min-samsung_padding, samsung_max+samsung_padding, samsung_diff/4))

#     plt.subplot(1, 2, 2)
#     plt.plot(range(len(hyundai_col)), np.array(hyundai_col[0:len(hyundai_col)]))
#     plt.yticks(np.arange(hyundai_min-hyundai_padding, hyundai_max+hyundai_padding, hyundai_diff/4))

# plt.show()


solve=0

x1=np.array(samsung)
x2=np.array(hyundai)
y=hyundai[hyundai.columns[solve]]
print(x1.shape,x2.shape)
# plt.plot(range(len(y)),y)
# plt.show()
ts=19
def split_and_scaling(x,ts):
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    x[4:]=scaler.fit_transform(x[4:])
    def split_to_time(data,ts):
        gen = (data[i:i+ts]for i in range(len(data)-ts+1))
        return np.array(list(gen))
    x=split_to_time(x,ts)
    x_train=x[:-2]
    x_test=np.reshape(x[-1],[1]+list(x_train.shape[1:]))
    print(x_train.shape)
    return x_train,x_test
x1_train,x1_test=split_and_scaling(x1,ts)
x2_train,x2_test=split_and_scaling(x2,ts)

y_train=y[ts+1:]


# 2. model build
input1=Input(shape=(x1_train.shape[1:]))
input2=Input(shape=(x2_train.shape[1:]))
merge=Concatenate()((input1,input2))
layer=LSTM(32)(merge)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
layer=Dense(16,activation='linear')(layer)
output=Dense(1)(layer)
model=Model(inputs=(input1,input2),outputs=output)
model.summary()



# 3. compile, training
from tensorflow.python.keras.callbacks import EarlyStopping 
import time
model.compile(loss='mse',optimizer='adam')
start_time=time.time()
x1_val,x2_val,y_val=x1_train[4*len(y_train)//5:],x2_train[4*len(y_train)//5:],y_train[4*len(y_train)//5:]
model.fit([x1_train,x2_train],y_train
          ,epochs=500,batch_size=len(x1_train)//40
          ,verbose=True,validation_data=([x1_val,x2_val],y_val)
          ,callbacks=EarlyStopping(monitor='val_loss',mode='min',patience=100,verbose=True,restore_best_weights=True))

# 4. predict
from sklearn.metrics import r2_score
evl=str()
evl+=f'구하는 값 : {samsung.columns[solve]}\n'
evl+=f'직전값 : {y_train[-1]} 예측값:{round(float(model.predict([x1_test,x2_test],batch_size=200,verbose=True)[0,0]),2)}\n'
y_pred=model.predict([x1_val,x2_val],batch_size=200,verbose=True)
evl+=f'결정계수 : {r2_score(y_val,y_pred)}\n'
evl+=f'런타임 : {round(time.time()-start_time,2)} 초\n'

print(evl)


x1_val,x2_val,y_val=x1_train,x2_train,y_train
y_pred=model.predict([x1_val,x2_val],batch_size=200,verbose=True)
plt.plot(range(len(y_val)),y_val,label='real')
plt.plot(range(len(y_val)),y_pred,label='model')
plt.legend()
plt.show()

# 5. save
model.save_weights('./_save/samsung/keras53_samsung4_jsw.h5')