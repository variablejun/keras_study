
from re import M
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input, Dropout, Conv1D
import numpy as np 

from sklearn.datasets import load_breast_cancer
datasets = load_breast_cancer()

#1.data
x = datasets.data
y = datasets.target
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

'''

#model = load_model('./_save/ModelCheckpoint/keras47_Model_Save.h5')
model = load_model('./_save/ModelCheckpoint/keras47_MCP.hdf5')
import time
starttime = time.time()
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime
print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
기존
걸린시간 0.056778907775878906
loss :  0.6150440573692322
accuracy :  0.6634958386421204

savemodel
걸린시간 1.4933924674987793
loss :  0.6150440573692322
accuracy :  0.6634958386421204

MCP
걸린시간 1.4878907203674316
loss :  0.5971330404281616
accuracy :  0.6872770190238953
'''