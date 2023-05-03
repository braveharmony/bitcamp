from sklearn.preprocessing import PolynomialFeatures
import numpy as np
x=np.arange(12).reshape((4,3))
print(x)
print(PolynomialFeatures(degree=3).fit_transform(x).shape)
