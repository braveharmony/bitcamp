import numpy as np
from hyperopt import hp,Trials,tpe,fmin,STATUS_OK


search_space={
    'learning_rate':hp.uniform('learning_rate',0.001,1),
    'max_depth':hp.quniform('max_depth',3,16,1),
    'num_leaves':hp.quniform('num_leaves',24,64,1),
    # 'min_child_samples':hp.quniform('min_child_samples',10,200,1),
    # 'min_child_weight':hp.uniform('min_child_weight',1,50),
    'subsample':hp.uniform('subsample',0.5,1),
    # 'colsample_bytree':hp.uniform('colsample_bytree',0.5,1),
    # 'max_bin':hp.quniform('max_bin',10,500,1),
    # 'reg_lambda':hp.uniform('reg_lambda',0.001,10),
    # 'reg_alpha':hp.uniform('reg_alpha',0.01,50)
}
# hp.quniform(label,low,high,q)
# hp.uniform(label,low,high)
# hp.randint(label,upper)
# hp.loguniform(label,low,hogh)
def y_function(*args,**para):
    # para['max_depth']=int(para['max_depth'])
    # para['num_leaves']=int(para['num_leaves'])
    # para['min_child_samples']=int(para['min_child_samples'])
    # para['max_bin']=int(para['max_bin'])

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
    
    from sklearn.metrics import r2_score,mean_squared_error
    return mean_squared_error(y_test,model.predict(x_test))

trial_val=Trials()
best=fmin(
    fn=y_function,
    space=search_space,
    algo=tpe.suggest,
    trials=trial_val,
    max_evals=100,
    # rstate=np.random.default_rng(seed=10)
)
print(best)