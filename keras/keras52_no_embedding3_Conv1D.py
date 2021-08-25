#Conv1D사용
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
[[2, 3], [1, 4], [1, 5, 6], [7, 8, 9], [10, 11, 12, 13, 14], [15], [16], [17, 18], [19, 20], [21], [2, 22], [1, 23], [24, 25, 26]]
크기가 다른데이터셋에 0을 채워 맞춰주는것
0을 앞에 채우는 이유는 LSTM과 같은 RNN적용시 앞에데이터보다 뒤에 데이터가 가중치에 더 많은
영향을 주기 때문이다.
[[ 0  0  0  2  3]
 [ 0  0  0  1  4]
 [ 0  0  1  5  6]
 [ 0  0  7  8  9]
 [10 11 12 13 14]
 [ 0  0  0  0 15]
 [ 0  0  0  0 16]
 [ 0  0  0 17 18]
 [ 0  0  0 19 20]
 [ 0  0  0  0 21]
 [ 0  0  0  2 22]
 [ 0  0  0  1 23]
 [ 0  0 24 25 26]]
(13, 5)
maxlen이상의 큰 데이터는 앞에서 부터 버린다.
원핫인코딩은 데이터가 너무 크면 연산수가 기하급수적으로 증가한다 
그래서 1이 있는 곳이 의미있는 데이터이고 0은 의미가 없을가능성이 높다
그래서 원핫인코딩한 것을 2차원 백터로 수치화 시켜 수직선상에 놓는다.
-> 유사성을 찾을수있디.
'''

pad_x = pad_x.reshape(13,5,1)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Input,Conv1D
model = Sequential()
model.add(Conv1D(77, 2,input_shape=(5,1)))
model.add(LSTM(32))
model.add(Dense(1, activation= 'sigmoid'))
'''
model = Sequential()
model.add(Dense(77,input_shape=(5,)))
model.add(Flatten())
model.add(Dense(1, activation= 'sigmoid'))
model.summary()

함수형 모델 사용시
input_length를 찾아서 정확히 명시해주고 처음에 넣어준다 생략불가, 이름만생략
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
params 같다.
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
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
acc : [0.010841475799679756, 1.0]

acc : [0.19242896139621735, 0.9230769276618958]
'''