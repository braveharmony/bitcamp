# Linear Discriminant Analysis
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from xgboost import XGBClassifier
from sklearn.datasets import load_diabetes

x,y=load_diabetes(return_X_y=True)

print(x.shape)

lda=LDA(n_components=7)
x_lda=lda.fit_transform(x,y)
print(x_lda.shape)

#성호는 캘리포니아에서 라운드처리했어
#그러다보니 그거도 정수형이라서 클래스로 인식된거야
#그래서 돌아간거야

#회귀데이터는 원칙적으로 에러인데
#위처럼 돌리고싶으면 돌려도 될?몰?루

