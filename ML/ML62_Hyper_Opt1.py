import hyperopt
from hyperopt import hp,fmin,tpe,Trials,STATUS_OK
search_space={
    'x1':hp.quniform('x',-10,10,1),
    'x2':hp.quniform('x2',-15,15,1)
}
def objective_func(ss):
    x1=ss['x1']
    x2=ss['x2']
    return x1**2-20*x2

trai_val=Trials()
best=fmin(fn=objective_func,
          space=search_space,
          algo=tpe.suggest,
          max_evals=10,
          trials=trai_val)
import pandas as pd
print(trai_val.results)
print(trai_val.vals)
pd.DataFrame(trai_val.vals).to_csv('./ML/trai_val.csv',index=False)
data=trai_val.vals.copy()
data['loss']=[i['loss'] for i in trai_val.results]
print(pd.DataFrame(data))