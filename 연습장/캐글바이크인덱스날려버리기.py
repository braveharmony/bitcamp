import pandas as pd
for i in range(10):
    path=f'./_save/kaggle_bike/03_14/forsub{i}.csv'
    df=pd.read_csv(path,index_col=0)
    df.to_csv(path,index=False)
#겁나 위험한 코드니 조심히 써야함