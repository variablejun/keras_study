from tensorflow.keras.datasets import imdb
import numpy as np
(x_train, y_train),(x_test,y_test) = imdb.load_data(num_words=10000)#제공하는 자료는 나눠져있다.

# print(x_train[0], type(x_train[0]))  <class 'list'>

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

'''
(25000,) (25000,)
(25000,) (25000,)
'''
print(len(x_train[0]), len(x_train[1])) # 218 189
print("최대길이 : ", max(len(i) for i in x_train))  #2494
print("평균길이 : ", sum(map(len, x_train))/ len(x_train)) #238.71364

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
x_train = pad_sequences(x_train, maxlen=2000,padding='pre')
x_test = pad_sequences(x_test, maxlen=2000,padding='pre')
print(x_test.shape, x_train.shape)
print(type(x_test), type(x_train[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'> 패딩하면바뀜
print(x_train[0]) #[  0   0   0 ...  19 178  32]

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(y_test.shape, y_train.shape)#(25000, 2) (25000, 2)

print(np.unique(x_train))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM


model = Sequential()     
model.add(Embedding(input_dim=25000, output_dim=55,input_length=None))
model.add(LSTM(32))
model.add(Dense(2, activation= 'softmax'))

model.compile(loss='categorical_crossentropy',optimizer = 'adam', metrics=['acc'])
model.fit(x_train,y_train,epochs=10,batch_size=256,validation_split=0.3)

loss = model.evaluate(x_train,y_train)

print('Loss :' , loss[0])
print('Acc', loss[1])
'''
softmax category
Loss : 0.17248986661434174
Acc 0.9516000151634216
'''
