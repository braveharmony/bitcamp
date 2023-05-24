import numpy as np


bayesian_params={
    'max_depth':(3,16),
    'num_leaves':(24,64),
    'min_child_samples':(10,200),
    'min_child_weight':(1,50),
    'subsample':(0.5,1),
    'colsample_bytree':(0.5,1),
    'max_bin':(10,500),
    'reg_lambda':(0.001,10),
    'reg_alpha':(0.01,50)
}

def y_function(**para):
    para['max_depth']=int(para['max_depth'])
    para['num_leaves']=int(para['num_leaves'])
    para['min_child_samples']=int(para['min_child_samples'])
    para['max_bin']=int(para['max_bin'])

    from sklearn.datasets import load_diabetes
    x,y=load_diabetes(return_X_y=True)
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=0)
    from sklearn.preprocessing import RobustScaler
    scaler=RobustScaler()
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    
    from lightgbm import LGBMRegressor
    model=LGBMRegressor(**para)
    model.fit(x_train,y_train)
    
    from sklearn.metrics import r2_score
    return r2_score(y_test,model.predict(x_test))


from bayes_opt import BayesianOptimization

optimizer=BayesianOptimization(
    f=y_function,
    pbounds=bayesian_params,
    random_state=0,
    
)

optimizer.maximize(init_points=2,n_iter=100)
print(optimizer.max)