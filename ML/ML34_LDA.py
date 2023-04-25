# Linear Discriminant Analysis
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import numpy as np
from xgboost import XGBClassifier
from tensorflow.keras.datasets import cifar10


x,y=load_iris(return_X_y=True)
lda=LDA(n_components=2)
lda.fit(x,y)
x=lda.transform(x)
model=XGBClassifier(tree_method='gpu_hist',
                    predictor='gpu_predictor',
                    gpu_id=0,n_estimators=100,learning_rate=0.3,
                    max_depth=4)
model.fit(x,y)
print(model.score(x,y))

pred=model.predict(x)

import matplotlib.pyplot as plt
# 각 y값에 대해 lda된 x의 시각화
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
for i, label in enumerate(('setosa', 'versicolor', 'virginica')):
    plt.scatter(x[y==i, 0], x[y==i, 1], label=label)

plt.legend(loc='best')
plt.title("LDA Transformed Features for Iris Dataset")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")

# 각 y값에 대해 XGBClassifier로 분류된 x의 시각화
plt.subplot(1,2,2)
for i, label in enumerate(('setosa', 'versicolor', 'virginica')):
    plt.scatter(x[pred==i, 0], x[pred==i, 1], label=label)

plt.legend(loc='best')
plt.title("XGBClassifier Predictions for Iris Dataset")
plt.xlabel("LDA Component 1")
plt.ylabel("LDA Component 2")
plt.show()