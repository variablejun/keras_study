# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
x_train = np.load('./_save/_npy/k55_x_data_fashion_mnist_x_train.npy')
y_train = np.load('./_save/_npy/k55_y_data_fashion_mnist_y_train.npy')
x_test = np.load('./_save/_npy/k55_x_data_fashion_mnist_x_test.npy')
y_test = np.load('./_save/_npy/k55_y_data_fashion_mnist_y_test.npy')

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

from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=3)
reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,factor=0.5) 

import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=128, validation_split=0.003, verbose=2,callbacks=[es,reduce_LR]) 

loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime
print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''

걸린시간 129.6283552646637
loss :  0.6326466202735901
accuracy :  0.855400025844574

Reduce 적용후
0.1
걸린시간 56.168118476867676
loss :  2.3027114868164062
accuracy :  0.10000000149011612
0.01
loss :  2.3025572299957275
accuracy :  0.10000000149011612
0.001
loss :  0.659654438495636
accuracy :  0.8639000058174133
'''