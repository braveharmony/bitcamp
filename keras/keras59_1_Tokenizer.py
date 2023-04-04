from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

text = '나는 진짜 매우 매우 맛있는 밥을 엄청 마구 마구 마구 먹었다.'
token=Tokenizer()
token.fit_on_texts([text])
print(token.word_index)
print(token.word_counts)
print(np.array(token.texts_to_sequences([text])).shape)
from tensorflow.keras.utils import to_categorical
print(to_categorical(token.texts_to_sequences([text])))

import pandas as pd
print(np.array(pd.get_dummies(token.texts_to_sequences([text])[0],prefix='number')))

from sklearn.preprocessing import OneHotEncoder
print(OneHotEncoder().fit_transform(np.array(token.texts_to_sequences([text])).T).toarray())