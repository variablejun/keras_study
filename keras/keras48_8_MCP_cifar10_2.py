#모델링완성 하시오 32,32,3
# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
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
model.add(Conv1D(128,2,activation = 'relu',input_shape=(32,96)))
model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(10, activation='sigmoid')) 

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=3)
cp = ModelCheckpoint(monitor='val_accuracy', save_best_only=True,mode='auto', filepath='./_save/ModelCheckpoint/keras47_MCP.hdf5')

import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=256, validation_split=0.003, verbose=2,callbacks=[es,cp]) 
model.save('./_save/ModelCheckpoint/keras47_Model_Save.h5')

loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
 epochs=20
걸린시간 16.315703868865967
loss :  2.3025879859924316
accuracy :  0.10000000149011612

epochs=100
걸린시간 60.926353216171265
loss :  3.089942216873169
accuracy :  0.47909998893737793

'''