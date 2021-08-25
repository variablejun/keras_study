from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

import numpy as np 
dataset = load_breast_cancer()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (569, 30) (569,) y 0 과 1로 구성

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
# (29, 30) (540, 30)
x_train = x_train.reshape(540,30,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(29,30, 1,1)

model = Sequential()
model.add(Conv2D(256,3 ,padding = 'same' ,input_shape=(30,1,1)))
model.add(Conv2D(128,3,padding = 'same' ,activation = 'relu'))

model.add(Conv2D(64,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(32,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(128,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(32,3,padding = 'same' ,activation = 'relu'))

model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류모델
'''
output layer의 반환값을 0~1사이에 값으로 만들주늗것
'''
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, callbacks=[es]) 
import time
starttime = time.time()
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime
print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
모델링수정전
loss :  0.6461238861083984
accuracy :  0.6551724076271057

모델링수정후
걸린시간 0.11479067802429199
loss :  0.15711985528469086
accuracy :  0.9655172228813171

2D
걸린시간 0.12848258018493652
loss :  0.001487557776272297
accuracy :  1.0

'''