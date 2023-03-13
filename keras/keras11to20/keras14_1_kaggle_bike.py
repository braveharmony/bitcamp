import tensorflow as tf
import numpy as np
import random, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LeakyReLU,Dropout
import tensorflow.keras.backend as K

def RMSE(y_test,y_predict):
    return np.sqrt(mean_squared_error(y_test,y_predict))

# 0. seed initialization
seed=0
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# 1. data
path='./_data/kaggle_bike/'
df=pd.read_csv(path+'train.csv')
# print(df.describe())
fig=df.columns
# print(df.isnull().sum())
x=df.drop([fig[0],fig[-3],fig[-2],fig[-1]],axis=1)
y=df[fig[-1]]

print(x.shape,y.shape)
seedset=[]
scoreset=[]
firstlayerset=[]
secondlayerset=[]
thirdlayerset=[]
epochset=[]
index_feature=['seed','score','firstlayer','secondlayer','thirdlayer','epoch']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7,random_state=seed)

for firstlayer in range(1,5):
    for secondlayer in range(1,5):
        for thirdlayer in range(1,5):
            for epo in range(1,11):
                # 2. model build
                epoch=300*epo
                model=Sequential()
                model.add(Dense(2**firstlayer,input_dim=x.shape[1],activation='relu'))
                model.add(Dense(2**secondlayer,activation='relu'))
                model.add(Dense(2**thirdlayer,activation='relu'))
                model.add(Dense(1))


                # 3. compile,training
                model.compile(loss='mse',optimizer='adam')
                model.fit(x_train,y_train,batch_size=len(x),epochs=epoch)


                # 4. evaluate, predict
                loss=model.evaluate(x_test,y_test)
                y_predict=model.predict(x_test)
                rmse=RMSE(y_test,y_predict)
                print(f'loss : {loss} \n rmse : {rmse}')

                ######################### 데이터 저장 ##############################
                seedset.append(seed);scoreset.append(rmse);firstlayerset.append(firstlayer);secondlayerset.append(secondlayer);thirdlayerset.append(thirdlayer);epochset.append(epoch)
                inv=pd.DataFrame({index_feature[0]:seedset, index_feature[1]:scoreset,
                    index_feature[2]:firstlayerset,index_feature[3]:secondlayerset,index_feature[4]:thirdlayerset,
                    index_feature[5]:epochset
                })
                print(inv)
                inv.to_csv('./연습장/자동화펙터.csv',index=False)

# ################## 테스트csv가져와서 제출할 y만듬#################
# fortest=pd.read_csv(path+'test.csv')
# figtest=fortest.columns
# xfortest=fortest.drop([fig[0]],axis=1)
# # print(xfortest.shape)
# yfortest=model.predict(xfortest)
# # print(yfortest)

# # ##################서브밋csv 가져와서 y랑 마지함###################
# # submit=pd.read_csv(path+"sampleSubmission.csv")
# # submit[fig[-1]]=yfortest
# # # print(submit)
# # pathsave='./_save/kaggle_bike/'
# # submit.to_csv(pathsave+'mltestsample.csv',index=False)