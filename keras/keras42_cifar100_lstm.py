#overfit 극복
# 훈련데이터를 최대한 많이
# 노멀라이제이션(정규화) 사용
# 드롭아웃
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(50000, 32, 32*3) # 3차원
x_test = x_test.reshape(10000, 32,32*3)




'''

'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''
'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() 
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(10000, 100)
(50000, 100)
'''

model = Sequential() 

model = Sequential()
model.add(LSTM(512,activation = 'relu',input_shape=(32,96)))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(100, activation='sigmoid'))
'''

'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
StandardScaler
걸린시간 60.21109747886658
loss :  4.368515491485596
accuracy :  0.20600000023841858

걸린시간 84.6844265460968
loss :  4.014658451080322
accuracy :  0.20960000157356262

걸린시간 78.38208365440369
loss :  0.04740132763981819
accuracy :  0.2329999953508377

걸린시간 179.6003556251526
loss :  0.04681005701422691
accuracy :  0.2378000020980835


RNN
걸린시간 254.24207973480225
loss :  0.09794498234987259
accuracy :  0.009999999776482582

걸린시간 260.92588925361633
loss :  0.056008417159318924
accuracy :  0.009999999776482582

걸린시간 1034.4530498981476
loss :  0.0560016892850399
accuracy :  0.009999999776482582
'''