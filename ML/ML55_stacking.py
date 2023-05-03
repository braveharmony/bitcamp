from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from catboost import CatBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
x,y=load_breast_cancer(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,stratify=y)

scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

model=StackingClassifier(
    estimators=[('xg',XGBClassifier()),('LGBM',LGBMClassifier()),('lr',LogisticRegression()),('Knn',KNeighborsClassifier(8))],final_estimator=CatBoostClassifier(iterations=10,verbose=False)
)

model.fit(x_train,y_train)

print(model.score(x_test,y_test))