import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1000개의 랜덤한 값을 생성합니다.
x = np.random.randn(1000)

# # 히스토그램을 그립니다.
# plt.figure(0)
# sns.histplot(x, kde=True)

# # 상관 관계를 계산합니다.
data = np.random.randn(1000, 4)
corr = np.corrcoef(data, rowvar=False)

# # 상관 관계 히트맵을 그립니다.
# plt.figure(1)
# sns.heatmap(corr, annot=True)

# # 바이올린 그래프를 그립니다.
# data = np.random.randn(1000, 2)
# plt.figure(2)
# sns.violinplot(x=data[:,0], y=data[:,1])

# # 스트립 플롯을 그립니다.
df = pd.DataFrame({'experiment': ['Experiment A', 'Experiment A', 'Experiment B', 'Experiment C', 'Experiment C'], 
                   'measurement': [1.2, 2.5, 3.1, 0.8, 1.7]})

plt.figure(3)
sns.stripplot(x=df.columns[0], y=df.columns[1], data=df, jitter=True)

# 그래프를 보여줍니다
plt.show()
