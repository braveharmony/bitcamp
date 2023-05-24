from sklearn.datasets import load_breast_cancer,fetch_california_housing
from sklearn.metrics import accuracy_score,r2_score,mean_squared_error
from sklearn.ensemble import RandomForestClassifier,VotingClassifier,VotingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

x,y=load_breast_cancer(return_X_y=True)
x_train,x_test,y_train,y_test=train_test_split(
    x,y,train_size=0.8,stratify=y
)

scaler=StandardScaler()
x_train =scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

lr=LogisticRegression()
knn=KNeighborsClassifier(n_neighbors=8)
dt=DecisionTreeClassifier()

model=VotingClassifier(
    estimators=[('lr',lr),('knn',knn),('dt',dt)],
    voting='soft'
)

model.fit(x_train,y_train)
print(f'model score : {model.score(x_test,y_test)}\nvoting_acc : {accuracy_score(y_test,model.predict(x_test))}')
for reg in [lr,knn,dt]:
    reg.fit(x_train,y_train)
    print(f'{type(reg).__name__} acc : {accuracy_score(y_test,reg.predict(x_test))}')



# x,y=fetch_california_housing(return_X_y=True)

# x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)
# scaler=StandardScaler()
# x_train=scaler.fit_transform(x_train)
# x_test=scaler.transform(x_test)
# xg=XGBRegressor()
# lg=LGBMRegressor()
# cat=CatBoostRegressor(verbose=False)
# model=VotingRegressor(estimators=[('xg',xg),('lg',lg),('cat',cat)])

# model.fit(x_train,y_train)
# print(f'score : {model.score(x_test,y_test)}, 결정계수 : {r2_score(y_test,model.predict(x_test))}')

# for reg in [xg,lg,cat]:
#     reg.fit(x_train,y_train)
#     print(f'{type(reg).__name__} 결정계수 : {r2_score(y_test,reg.predict(x_test))}')
#     print(f'{type(reg).__name__} mse : {mean_squared_error(y_test,reg.predict(x_test))}')

