import numpy as np
import pandas as pd
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
def file_path()->str:
    return 'c:/forcon/_data/finedust/'
train_files=glob.glob(file_path()+'TRAIN/*.csv')
test_input_files=glob.glob(file_path()+'test_input/*.csv')
print(type(train_files[1]))
df=[]
for filename in train_files:
    df.append(pd.read_csv(filename))
    
train_dataset=pd.concat(df,axis=0,ignore_index=True)
df=[]
for filename in test_input_files:
    df.append(pd.read_csv(filename))
test_input_dataset=pd.concat(df,axis=0,ignore_index=True)
le=LabelEncoder()
train_dataset['locate']=le.fit_transform(train_dataset['측정소'])
test_input_dataset['location']=le.transform(test_input_dataset['측정소'])
train_dataset=train_dataset.drop(['측정소'],axis=1)
test_input_dataset=test_input_dataset.drop(['측정소'],axis=1)


def split_datetime(DataFrame:pd.DataFrame)->pd.DataFrame:
    DataFrame.insert(1, '월', DataFrame['일시'].str[:2].astype('int8'))
    DataFrame.insert(2, '일', DataFrame['일시'].str[3:5].astype('int8'))
    DataFrame.insert(3, '시', DataFrame['일시'].str[6:8].astype('int8'))
    DataFrame=DataFrame.drop(['일시'],axis=1)
    return DataFrame
train_dataset=split_datetime(train_dataset)
test_input_dataset=split_datetime(test_input_dataset)

y=train_dataset['PM2.5']
x=train_dataset.drop(['PM2.5'],axis=1)
print(train_dataset)
print(train_dataset.info())
y.fillna(y.mean(), inplace=True)
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,shuffle=True)


parameters={'n_estimators' : 10000,#디폴트 100,1~inf,int
            'learning_rate' : 0.06,#디폴트 0.3/0~1/float
            'max_depth':10,#디폴트 6/0~inf/int
            'gamma':1,#디폴트 0/0~inf/float
            'min_child_weight':1,#디폴트 1/0~inf/float
            'subsample':1,#디폴트 1/0~1/float
            'colsample_bytree':1,#디폴트 1/0~1/float
            'colsample_bylevel':1,#디폴트 1/0~1/float
            'colsample_bynode':1,#디폴트 1/0~1/float
            'reg_alpha':0,#디폴트 0/0~inf/float
            'reg_lambda':1,#디폴트 1/0~inf/float
            'random_state':0,
            # 'verbose':1
            }

from xgboost import XGBRegressor
model = XGBRegressor()
model.set_params(**parameters,eval_metric='mae',early_stopping_rounds=200)
import time
start_time=time.time()
model.fit(x_train,y_train,eval_set=((x_train,y_train),(x_test,y_test)),verbose=1)
runtime=np.round(time.time()-start_time,2)

from sklearn.metrics import r2_score,mean_absolute_error

y_predict=model.predict(x_test)
results=model.score(x_test,y_test)
r2=r2_score(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
print(f'runtime:{runtime}\nmodel score:{results}\n결정계수:{r2}\nmae:{mae}')