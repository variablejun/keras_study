from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000, test_split=0.2)


print(x_train[0], type(x_train[0]))

'''
[1, 2, 2, 8, 43, 10, 447, 5, 25, 207, 270, 5, 3095, 
111, 16, 369, 186, 90, 67, 7, 89, 5, 19, 102, 6, 19, 
124, 15, 90, 67, 84, 22, 482, 26, 7, 48, 4, 49, 8, 864, 
39, 209, 154, 6, 151, 6, 83, 11, 15, 22, 155, 11, 15, 7, 
48, 9, 4579, 1005, 504, 6, 258, 6, 272, 11, 15, 22, 134, 44, 
11, 15, 16, 8, 197, 1245, 90, 67, 52, 29, 209, 30, 32, 132, 6, 
109, 15, 17, 12] <class 'list'>
리스트는 크기 제한이 없어 배열의 방마다 크기가 다를 수 있다.
'''

print(len(x_train[0]), len(x_train[1])) # 87 56 리스트는 쉐입이 먹지 않는다, 어트리뷰트가없다.
print(x_test.shape, x_train.shape)
print(y_test.shape, y_train.shape)
#(2246,) (8982,)
#(2246,) (8982,)
print(type(x_test)) # <class 'numpy.ndarray'>
print("뉴스기사의 최대길이 : ", max(len(i) for i in x_train)) 

print("뉴스기사의 평균길이 : ", sum(map(len, x_train))/ len(x_train)) 
#뉴스기사의 최대길이 :  2376
#뉴스기사의 평균길이 :  145.5398574927633
'''
import matplotlib.pyplot as plt
plt.hist([len(s) for s in x_train], bins=50)
plt.show()
'''

# 전처리
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
x_train = pad_sequences(x_train, maxlen=100,padding='pre')
x_test = pad_sequences(x_test, maxlen=100,padding='pre')
print(x_test.shape, x_train.shape)
print(type(x_test), type(x_train[0]))
print(x_train[0])
'''
(2246, 100) (8982, 100)
<class 'numpy.ndarray'> <class 'numpy.ndarray'>
[   0    0    0    0    0    0    0    0    0    0    0    0    0    1
    2    2    8   43   10  447    5   25  207  270    5 3095  111   16
  369  186   90   67    7   89    5   19  102    6   19  124   15   90
   67   84   22  482   26    7   48    4   49    8  864   39  209  154
    6  151    6   83   11   15   22  155   11   15    7   48    9 4579
 1005  504    6  258    6  272   11   15   22  134   44   11   15   16
    8  197 1245   90   67   52   29  209   30   32  132    6  109   15
   17   12]

   y = 카테고리 46개 -> softmax연산

'''

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test.shape, y_train.shape)# (2246, 46) (8982, 46)

print(np.unique(x_train))
#모델링
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM


model = Sequential()     
model.add(Embedding(input_dim=10001, output_dim=55,input_length=None))
model.add(LSTM(32))
model.add(Dense(46, activation= 'softmax'))

#예측


model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=100,batch_size=32,validation_split=0.3)

loss = model.evaluate(x_train,y_train)

print('Loss :' , loss[0])
print('acc', loss[1])
'''
Loss : 0.8388067483901978
acc 0.874749481678009
'''

'''

'''