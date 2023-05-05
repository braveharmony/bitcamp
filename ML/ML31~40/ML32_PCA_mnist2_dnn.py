from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC
import pandas as pd
def reshaping(x:np.ndarray):
    return x.reshape(x.shape[0],-1)

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x=reshaping(np.concatenate((x_train,x_test),axis=0))
y=pd.get_dummies(np.concatenate((y_train,y_test),axis=0))
y_train=y[:len(y_train)];y_test=y[len(y_train):]
pca=PCA(n_components=x.shape[1])
pca.fit(x)
evr=np.cumsum(pca.explained_variance_ratio_)
numlist=[len(np.argwhere(evr>=0.95)),len(np.argwhere(evr>=0.99)),len(np.argwhere(evr>=0.999)),len(np.argwhere(evr>=1.0))]
print(f'0.95:{numlist[0]}')
print(f'0.99:{numlist[1]}')
print(f'0.999:{numlist[2]}')
print(f'1.0:{numlist[3]}')
def run_DNN(num,x,x_train,x_test,y_train,y_test):
    pca=PCA(n_components=x.shape[1]-num)
    x_PCA=pca.fit_transform(x)
    x_train=x_PCA[:len(x_train)];x_test=x_PCA[len(x_train):]
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense,Dropout,Input
    model=Sequential()
    model.add(Input(shape=x_train.shape[1:]))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(1/16))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(1/16))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(1/16))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(1/16))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(1/16))
    model.add(Dense(y_train.shape[1],activation='softmax'))

    from tensorflow.keras.callbacks import EarlyStopping as es
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')
    model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10000,batch_size=len(x_train)//100,
                callbacks=es(monitor='val_acc',mode='max',patience=50,restore_best_weights=True))
    from sklearn.metrics import accuracy_score
    return accuracy_score(y_test,np.round(model.predict(x_test)))
acclist=[]
for num in numlist:
    acclist.append(run_DNN(num,x,x_train,x_test,y_train,y_test))
print('원본DNN acc : ㅁ?ㄹ')
print('원본CNN acc : 1.0')
for i in range(len(numlist)):
    print(f'PCA {numlist[i]} : {acclist[i]}')