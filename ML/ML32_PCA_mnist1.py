from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA
import numpy as np
from sklearn.svm import SVC



def reshaping(x:np.ndarray):
    return x.reshape(x.shape[0],-1)

(x_train,y_train),(x_test,y_test)=mnist.load_data()
x=reshaping(np.concatenate((x_train,x_test),axis=0))
pca=PCA(n_components=x.shape[1])
pca.fit(x)
evr=np.cumsum(pca.explained_variance_ratio_)
print(f'0.95:{len(np.argwhere(evr>=0.95))}')
print(f'0.99:{len(np.argwhere(evr>=0.99))}')
print(f'0.999:{len(np.argwhere(evr>=0.999))}')
print(f'1.0:{len(np.argwhere(evr>=1.0))}')
