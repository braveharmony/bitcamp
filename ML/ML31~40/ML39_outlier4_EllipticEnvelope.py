import numpy as np
aaa= np.array([[-10,2,3,4,5,6,700,8,9,10,11,12,50],
               [100,200,-30,400,500,600,-70000,800,900,1000,210,420,350]])

def outliers_2D(data_2D:np.ndarray)->list:
    outlier_indices=[]
    from sklearn.covariance import EllipticEnvelope
    outliers=EllipticEnvelope(contamination=.1)
    for i,v in enumerate(data_2D):
        outlier_indices+=[[i,j] for j in np.where(outliers.fit_predict(v.reshape(-1,1))==-1)[0]]
    return outlier_indices

print(outliers_2D(aaa))