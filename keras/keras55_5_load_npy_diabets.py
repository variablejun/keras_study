'''

import numpy as np
x_data = np.load('./_save/_npy/k55_x_data_diabetes.npy')
y_data = np.load('./_save/_npy/k55_y_data_diabetes.npy')
print(x_data)
print(y_data)

print(x_data.shape,y_data.shape) #(442, 10) (442,)
'''
from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Conv1D, Flatten
# import pandas as pd

#1.data
datasets = load_diabetes()

x =  np.load('./_save/_npy/k55_x_data_diabetes.npy')
y =  np.load('./_save/_npy/k55_y_data_diabetes.npy')


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)



# (23, 10) (419, 10)
x_train = x_train.reshape(419,10,1) # 
x_test = x_test.reshape(23, 10, 1) # 

#2.모델 구성

model = Sequential()
model.add(Conv1D(32,2,activation = 'relu',input_shape=(10,1)))
model.add(Flatten())
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

print('loss : ', loss)
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
print("걸린시간",end)

'''


Epoch 00138: early stopping
1/1 [==============================] - 0s 20ms/step - loss: 4070.3704
loss :  4070.370361328125
r2score  0.4542359135536461
걸린시간 7.06217551231384
'''