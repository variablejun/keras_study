'''
import numpy as np
x_data = np.load('./_save/_npy/k55_x_data_boston.npy')
y_data = np.load('./_save/_npy/k55_y_data_boston.npy')
print(x_data)
print(y_data)
print(x_data.shape,y_data.shape) #(506, 13) (506,)
'''
from re import M
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Conv1D
import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x =  np.load('./_save/_npy/k55_x_data_boston.npy')
y = np.load('./_save/_npy/k55_y_data_boston.npy')
'''

'''
print(np.max(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

x_train = x_train.reshape(480,13,1) #
x_test = x_test.reshape(26, 13, 1)

'''

'''




model = Sequential()
model.add(Conv1D(16,2,activation = 'relu',input_shape=(13,1)))
model.add(LSTM(64))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.01)

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,factor=0.5) 

import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = optimizer)
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es,reduce_LR]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss)
y_pred = model.predict(x_test) 

# y_pred = scaler.transform(y_pred)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
기존 
걸린시간 9.140683650970459
loss :  19.98594856262207
r2score  0.8449983696257106

LR적용시 0.1 과 0.001은 안좋은 결과가 나옴
걸린시간 7.026266813278198
loss :  36.389591217041016
r2score  0.7177794042361029 

'''