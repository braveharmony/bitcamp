import pandas as pd
for i in range(30):
    path=f'kaggle_bike/03_10/forsub{i}.csv'
    df=pd.read_csv(path)
    y=df[df.columns[-1]]
    y=y.apply(lambda x: 0 if x < 0 else x)
    df[df.columns[-1]]=y
    df.to_csv(path,index=False)
