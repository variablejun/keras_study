#overfit 극복
# 훈련데이터를 최대한 많이
# 노멀라이제이션(정규화) 사용
# 드롭아웃
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Flatten, Conv1D
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
model.add(Conv1D(512,2,activation = 'relu',input_shape=(32,96)))
model.add(Conv1D(256,2,activation = 'relu',))
model.add(Conv1D(128,2,activation = 'relu',))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(100, activation='sigmoid'))
'''

'''

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
157/157 [==============================] - 0s 3ms/step - loss: 0.0450 - accuracy: 0.1894
걸린시간 61.876680850982666
loss :  0.0450081005692482
accuracy :  0.18940000236034393

epochs=100
157/157 [==============================] - 0s 3ms/step - loss: 0.0529 - accuracy: 0.2036
걸린시간 139.4875614643097
loss :  0.05289130285382271
accuracy :  0.20360000431537628


'''