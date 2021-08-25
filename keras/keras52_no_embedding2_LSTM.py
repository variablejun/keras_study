from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

docs= ['너무 재미있어요', '참 최고에요', '참 잘만든 영화에요'
, '추천하고 싶은 영화입니다.','한 번 더 보고 싶네요','글쎄요',
 '별로에요','생각보다 지루해요','연기가 어색해요','제미없어요', '너무 재미없다','참 재밋네요','청순이가 잘생기긴 했어요']

 #긍 1 부 0
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
'''
{'참': 1, '너무': 2, '재미있어요': 3, '최고에요': 4, '잘만든': 5, '영화에요': 6, '추천하고': 7, '싶은': 8, '영화입니다': 9, '한': 10, '번': 11, '더': 12, '보고': 13, '싶네요': 14, '글쎄요': 15, '별로에요': 16, '생각보다': 17, '지루해요': 18, '연기가': 19, '어색해요': 20, '제미없어요': 21, '재미없다': 
22, '재밋네요': 23, '청순이가': 24, '잘생기긴': 25, '했어요': 26}
'''
print(token.word_index)
x = token.texts_to_sequences(docs)
print(x)
from tensorflow.keras.preprocessing.sequence import pad_sequences
pad_x = pad_sequences(x, padding='pre', maxlen=5)
print(pad_x)
print(pad_x.shape)

word_Size = len(token.word_index)
print(word_Size) #26
print(np.unique(pad_x)) # 0~ 26 -> input_dim=27
#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26]
'''

'''
pad_x = pad_x.reshape(13,5,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM

model = Sequential()
model.add(LSTM(77,input_shape=(5,1)))
model.add(Dense(1, activation= 'sigmoid'))
model.summary()
'''
DNN = units
CNN = filter
RNN = units
Embedding= output_dim

param = input_dim * output_dim 
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5, 77)             2079
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
Total params: 16,192
Trainable params: 16,192
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='binary_crossentropy',optimizer = 'adam', metrics=['acc'])
model.fit(pad_x,labels,epochs=100,batch_size=32)

acc = model.evaluate(pad_x,labels)
print('acc :',acc)

'''
Function call stack:
train_function -> train_function
acc : [0.00020239950390532613, 1.0]

acc : [0.25799593329429626, 0.9230769276618958]

'''