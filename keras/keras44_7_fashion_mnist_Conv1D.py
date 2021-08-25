# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(60000, 28,28* 1)
x_test = x_test.reshape(10000, 28,28*1)

'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''
ValueError: Expected 2D array, got 1D array instead:
array=[7 2 1 ... 4 5 6].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample
'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''

'''

model = Sequential()
model.add(Conv1D(32,2,activation = 'relu',input_shape=(28,28)))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime
print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
걸린시간 16.219507455825806
loss :  2.3025951385498047
accuracy :  0.10000000149011612

'''