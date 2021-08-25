import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0) # ./ 현재폴더 ../ 상위폴더 데이터 구분자 ;
# index는 없고 헤더는 첫번째 라인
datasets =datasets.values

print(datasets.shape) # (4898, 12)

x = datasets[:,0:11]
y = datasets[:,[11]]


'''

'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
'''

'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.9995, random_state=66)

x_train = x_train.reshape(4895,11,1)
x_test = x_test.reshape(3, 11,1)

'''
기존의 코드는 iloc으로 데이터프레임을 잘라 스케일러에서 형변환을 해주엇으나
스케일러가 오류나 스케일러를 뺏더니 데이터프레임이라서 reshape가 먹지않아
values로 형변환하여 사용

(4898, 12)
(3, 11) (4895, 11)

'''
model = Sequential()
model.add(LSTM(32,activation = 'relu',input_shape=(11,1)))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=100, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.0003, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
Epoch 00102: early stopping
1/1 [==============================] - 0s 96ms/step - loss: 0.8095 - accuracy: 1.0000
loss :  0.8094513416290283
accuracy :  1.0

1/1 [==============================] - 0s 16ms/step - loss: 0.8022 - accuracy: 1.0000
loss :  0.8022322654724121
accuracy :  1.0
'''



