import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Conv1D, Flatten
dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (569, 30) (569,) y 0 과 1로 구성
# 원핫 인코딩
'''

'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

x_train = x_train.reshape(142,4,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(8, 4, 1)
'''
(150, 4) (150,)
(8, 4) (142, 4)
'''
model = Sequential()
model.add(Conv1D(64,2,activation = 'relu',input_shape=(4,1)))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(3 ,activation='softmax'))
'''

'''
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_accuracy', save_best_only=True,mode='auto', filepath='./_save/ModelCheckpoint/keras47_MCP.hdf5')

import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.25, verbose=2,callbacks=[es, cp]) 
model.save('./_save/ModelCheckpoint/keras47_Model_Save.h5')

loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
Epoch 00021: early stopping
1/1 [==============================] - 0s 20ms/step - loss: 0.5361 - accuracy: 1.0000
걸린시간 2.6931183338165283
loss :  0.5360785722732544
accuracy :  1.0
'''