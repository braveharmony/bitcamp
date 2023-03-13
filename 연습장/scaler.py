from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
palette = "#008080"; sns.set_style('whitegrid')

# 원본:-10부터 10까지 랜덤 데이터 100개
a = [np.random.randint(-10, 10) for _ in range(100)]
a = np.array([a]).T


# 원본 플롯
plt.figure(0)
sns.lineplot(x=np.arange(len(a)), y=a.ravel(), palette=[palette])
# plt.plot(a, c=palette)
plt.title('original data')


# 모든 스케일러는 linear scaling을 기본 원칙으로 한다!
plt.figure(1)

# MinMaxScaler
# 최대값이 1, 최소값이 0이 되도록 스케일링!
MMS = MinMaxScaler()
MMS.fit(a)
aMMS = MMS.transform(a)

plt.subplot(2, 2, 1)
sns.lineplot(x=np.arange(len(a)),y=aMMS.ravel(), palette=[palette])
plt.title('MinMaxScaler', fontsize=10)
plt.ylabel(f'min : {np.min(aMMS)} max : {np.max(aMMS)}')

# MaxAbsScaler
# 절대값이 가장 큰 수의 절대값이 1이 되도록 스케일링!
MAS = MaxAbsScaler()
MAS.fit(a)
aMAS = MAS.transform(a)

plt.subplot(2, 2, 2)
sns.lineplot(x=np.arange(len(a)),y=aMAS.ravel(), palette=[palette])
plt.title('MaxAbsScaler', fontsize=10)
plt.ylabel(f'max(np.abs) : {np.max(np.abs(aMAS))}')

# StandardScaler
# 평균이 0, 분산이 1이 되도록 스케일링!
SS = StandardScaler()
SS.fit(a)
aSS = SS.transform(a)

plt.subplot(2, 2, 3)
sns.lineplot(x=np.arange(len(a)),y=aSS.ravel(), palette=[palette])
plt.title('StandardScaler', fontsize=10)
plt.ylabel(f'mean : {np.mean(aSS)} var : {np.var(aSS)}')

# RobustScaler
# 중앙값이 0, IQR(상,하위 25%의 차이)이 1이 되도록 스케일링!
RS = RobustScaler()
RS.fit(a)
aRS = RS.transform(a)

plt.subplot(2, 2, 4)
sns.lineplot(x=np.arange(len(a)),y=aRS.ravel(), palette=[palette])
plt.title('RobustScaler', fontsize=10)
plt.ylabel(f'median : {np.median(aRS)}  IQR : {np.percentile(aRS, 100 - 25) - np.percentile(aRS, 25)}')


# 마지막으로 겹쳐보면
plt.figure(3)
plt.plot(aMMS,label='MinMaxScaler')
plt.plot(aMAS,label='MaxAbsScaler')
plt.plot(aSS,label='StandardScaler')
plt.plot(aRS,label='RobustScaler')
plt.legend()

# 위에 그래프들 전부 출력
plt.show()
