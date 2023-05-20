# Linear Discriminant Analysis
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.datasets import cifar100

(x_train,y_train),(x_test,y_test)=cifar100.load_data()
x=np.reshape(np.concatenate((x_train,x_test),axis=0),(sum(map(len,(x_train,x_test))),-1))
y=np.reshape(np.concatenate((y_train,y_test),axis=0),(-1,))
print(x.shape)

lda=LDA(n_components=99)
x_lda=lda.fit_transform(x,y)
print(x_lda.shape)