# 실습
# 1~100까지의 데이터를
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Conv1D, Flatten
import numpy as np 

'''
x          y
1 2 3 4 5  6
'''
import numpy as np
x_data = np.array(range(1,101))
x_predict = np.array(range(96,105))
'''
96 ~ 100 ?
...
101~ 105 ?
'''
size = 56

def split_x(dataset, size):
     aaa=[]
     for i in range(len(dataset) - size + 1):
          subset = dataset[i : (i + size)]
          aaa.append(subset)
     return np.array(aaa)
dataset = split_x(x_data, size)
x = dataset[:, :5]
y = dataset[:,5]
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

x_train = x_train.reshape(42,5,1)
x_test = x_test.reshape(3,5,1)
x_predict = x_predict.reshape(9,1,1)
'''

(42, 5) (3, 5)
(42,) (3,)
(9,)
(9, 5) (86, 5)
(9,)
(9,)
'''
model = Sequential()
model.add(Conv1D(16,2,activation = 'relu',input_shape=(5,1)))
model.add(LSTM(64))
#model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime




print("걸린시간", end)
print('loss : ', loss)

'''
1/1 [==============================] - 0s 19ms/step - loss: 3.8747
걸린시간 3.691469192504883
loss :  3.8747012615203857

모델링 수정후
Epoch 00137: early stopping
1/1 [==============================] - 0s 21ms/step - loss: 0.0093
걸린시간 6.069395303726196
loss :  0.009300143457949162

컨보루션1D와 LSTM을 둘다 사용하여 3차원 데이터를 처리할수있는데
컨보루션이 빠르다.
'''