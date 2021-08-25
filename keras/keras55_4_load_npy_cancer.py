'''

import numpy as np
x_data = np.load('./_save/_npy/k55_x_data_breast_cancer.npy')
y_data = np.load('./_save/_npy/k55_y_data_breast_cancer.npy')
print(x_data)
print(y_data)

print(x_data.shape,y_data.shape) #(569, 30) (569,)

'''
from re import M
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Dropout, Conv1D
import numpy as np 

from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

#1.data
x = np.load('./_save/_npy/k55_x_data_breast_cancer.npy')
y = np.load('./_save/_npy/k55_y_data_breast_cancer.npy')
'''

'''
print(np.max(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

x_train = x_train.reshape(540,30,1) 
x_test = x_test.reshape(29,30, 1)

'''

'''
model = Sequential()
model.add(LSTM(35,activation = 'relu',return_sequences=True,input_shape=(30,1)))
model.add(Conv1D(32,2,activation = 'relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.003, callbacks=[es]) 
import time
starttime = time.time()
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime
print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''

Epoch 00033: early stopping
1/1 [==============================] - 0s 37ms/step - loss: 0.4094 - accuracy: 0.8763
걸린시간 0.05480384826660156
loss :  0.4093622863292694
accuracy :  0.8763377070426941
'''