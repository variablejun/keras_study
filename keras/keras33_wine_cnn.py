import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
#from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D, Dropout, GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0) # ./ 현재폴더 ../ 상위폴더 데이터 구분자 ;
# index는 없고 헤더는 첫번째 라인


print(datasets.shape) # (4898, 12)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]


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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

x_train = x_train.reshape(4895,11,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(3, 11, 1,1)

'''
(4898, 12)
(3, 11) (4895, 11)

'''
model = Sequential() 
model.add(Conv2D(256,3 ,padding = 'same' ,input_shape=(11,1,1)))
model.add(Conv2D(128,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(32,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(32,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
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
'''



