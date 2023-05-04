import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import QuantileTransformer

x,y=make_blobs(random_state=0,n_samples=50,centers=2,cluster_std=1)
print(x.shape,y.shape)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()