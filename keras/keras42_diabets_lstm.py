from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)



# (23, 10) (419, 10)
x_train = x_train.reshape(419,10,1) # 
x_test = x_test.reshape(23, 10, 1) # 

#2.모델 구성

model = Sequential()
model.add(LSTM(32,activation = 'relu',input_shape=(10,1)))
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

2D
Conv2D(32
loss :  3936.20166015625
r2score  0.4722255363438762
걸린시간 8.421845436096191

Conv2D(16
loss :  3744.376708984375
r2score  0.4979458503797921
걸린시간 6.6011621952056885

RNN
loss :  7280.37158203125
r2score  0.02383201959299286
걸린시간 16.764601230621338

튜닝

loss :  6811.85107421875
r2score  0.08665224824808326
걸린시간 14.23827075958252

'''