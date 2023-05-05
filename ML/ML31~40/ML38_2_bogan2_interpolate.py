import numpy as np
import pandas as pd
from datetime import datetime

dates=pd.to_datetime([f'4/{i}/2023'for i in range(25,31)])
print(dates)
print(type(dates))
print('=================================')
ts=pd.Series([2,np.nan,np.nan,8,10,np.nan],dates)
print(ts)
print('=================================')
ts=ts.interpolate()
print(ts)