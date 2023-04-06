from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences  
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import itertools
docs= ['너무 재밋어요','참 최고에요','참 잘 만든 영화에요',
       '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요',
       '별로에요','생각보다 지루해요','연기가 어색해요',
       '재미없어요','너무 재미없다','참 재밋네요','환희가 잘 생기긴 했어요',
       '환희가 안해요'
       ]

# 긍정 1 , 부정 0
labels=np.array([1,1,1,1,1,0,0,0,0,0,0,1,1,0])
# docs_with_labels=[[docs[i],labels[i]]for i in range(len(labels))]
x_predict='나는 성호가 정말 재미없다 너무 정말'

token=Tokenizer()
token.fit_on_texts(docs+[x_predict])
print(token.word_index)
print(token.word_counts)

x=token.texts_to_sequences(docs)

onehot=OneHotEncoder().fit(pad_sequences(token.texts_to_sequences(docs+[x_predict]),padding='pre',maxlen=8).reshape(-1,1))

def pad_and_encoding(x,onehot):
    pad_x=pad_sequences(x,padding='pre',maxlen=8)
    return np.array([onehot.transform(pad_x[i].reshape(-1,1)).toarray() for i in range(len(pad_x))]),onehot
x_pad,onehot=pad_and_encoding(x,onehot)

# from tensorflow.keras.preprocessing.sequence import pad_sequences
# x_pad=pad_sequences(x,padding='pre',maxlen=max((len(i)for i in x)))
# x_pad=x_pad.reshape(*x_pad.shape,1)

# 2. model build
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,Dropout,Input,LeakyReLU,Embedding,LSTM,Reshape,Conv1D
model=Sequential()
model.add(Input(x_pad.shape[1:]))
model.add(Conv1D(32,3))
model.add(Conv1D(32,3))
model.add(Conv1D(32,3))
model.add(Conv1D(32,2))
model.add(Flatten())
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(64,activation=LeakyReLU(0.5)))
model.add(Dropout(0.125))
model.add(Dense(1,activation='sigmoid'))

# model=Sequential()
# model.add(Input(x_pad.shape[1:]))
# model.add(Embedding())
# model.add(Dense(16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(16,activation='relu'))
# model.add(Dense(1,activation='sigmoid'))
# model.summary()

# 3. compile,training
model.compile(loss='binary_crossentropy',optimizer='adam',metrics='acc')
model.fit(x_pad,labels,batch_size=50,epochs=100)

# 4. 평가 예측
acc = model.evaluate(x_pad,labels)[1]
print('acc : ',acc)

x_predict='나는 성호가 정말 재미없다 너무 정말'
x_predict,onehot=pad_and_encoding(token.texts_to_sequences([x_predict]),onehot)
print(f'y_predict : {model.predict(x_predict)}')
