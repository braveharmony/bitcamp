import pandas as pd
import numpy as np
import tensorflow as tf
import random
from sklearn.preprocessing import LabelEncoder

load_path = './_data/calorie/'
save_path = 'D:/study_data/_save/dacon_calories/'

##########################################################################################################

def seed_initialization(seed: int) ->None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    return

##########################################################################################################

def load_data(x: str = 'train') -> pd.DataFrame:
    """
    주어진 키워드에 해당하는 CSV 파일을 불러와서 DataFrame으로 반환합니다.

    Args:
        x (str, optional): CSV 파일의 종류를 지정하는 키워드.\
    'train', 'test', 'sub' 중 하나를 사용할 수 있으며 기본값은 'train'입니다.
            
    Returns:
        pd.DataFrame: 불러온 CSV 파일에 해당하는 DataFrame
    """
    if x == 'train':
        return pd.read_csv(f'{load_path}train.csv',index_col=0)
    elif x == 'test':
        return pd.read_csv(f'{load_path}test.csv',index_col=0)
    elif x == 'sub':
        return pd.read_csv(f'{load_path}sample_submission.csv')
    else:
        raise ValueError('Input keyword is not allowed. Please enter "train", "test", or "sub".')

##########################################################################################################

def load_x_y(train_data:pd.DataFrame=load_data('train'),test_data:pd.DataFrame=load_data('test'),returnXy:bool=True):
    """
    판다스 데이터 프레임 train_data랑 test_data를 가져다가.
    Weight_Status열과 Gender열을 각각 라벨 인코딩
    """
    incoder={'Weight_Status':LabelEncoder(),'Gender':LabelEncoder()}
    
    train_data['Weight_Status']=incoder['Weight_Status'].fit_transform(train_data['Weight_Status'])
    test_data['Weight_Status']=incoder['Weight_Status'].transform(test_data['Weight_Status'])

    train_data['Gender']=incoder['Gender'].fit_transform(train_data['Gender'])
    test_data['Gender']=incoder['Gender'].transform(test_data['Gender'])
    if returnXy==True:return train_data.drop('Calories_Burned',axis=1),train_data['Calories_Burned'],test_data
    else: return test_data
##########################################################################################################

def save_data(x: pd.DataFrame) -> None:
    if not isinstance(x, pd.DataFrame):
        raise ValueError('Please enter a pandas DataFrame.')
    import datetime     
    now = datetime.datetime.now().strftime('%m월%d일 %H시%M분%S초')
    x.to_csv(f"{save_path}{now}세이브파일.csv",index=False)
    return
    
##########################################################################################################
