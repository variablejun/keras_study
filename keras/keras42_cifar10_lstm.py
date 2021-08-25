#모델링완성 하시오 32,32,3
# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
print(x_train.shape,y_train.shape ) 
print(x_test.shape,y_test.shape )
x_train = x_train.reshape(50000, 32, 32* 3)
x_test = x_test.reshape(10000, 32, 32*3)


'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
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
y_test = OE.transform(y_test).toarray() 
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(10000, 10)
(50000, 10)
'''


model = Sequential()
model.add(LSTM(128,activation = 'relu',input_shape=(32,96)))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10, activation='sigmoid')) 

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=256, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
StandardScaler
loss :  3.5586183071136475
accuracy :  0.5202999711036682

loss :  1.6345748901367188
accuracy :  0.5428000092506409

loss :  2.0461318492889404
accuracy :  0.5519999861717224

oss :  2.255540370941162
accuracy :  0.5586000084877014

RobustScaler
loss :  3.2573187351226807
accuracy :  0.5127000212669373

loss :  2.0250747203826904
accuracy :  0.546999990940094

RNN
걸린시간 263.8860836029053
loss :  2.3025999069213867
accuracy :  0.10000000149011612

걸린시간 260.55508756637573
loss :  2.3025872707366943
accuracy :  0.10000000149011612
'''