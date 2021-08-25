import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (569, 30) (569,) y 0 과 1로 구성
# 원핫 인코딩
'''
y의 데이터가 0, 1, 2 다중분류일때사용

0 -> [1, 0, 0]
1 -> [0, 1, 0]
2 -> [0, 0, 1]
3개의 값이 차이가 없도록 데이터가 존재하는 곳에 1을 넣어서 배열로 바꾸어준다
그러면 y의 쉐입은 150,3 으로 변하고 output layer의 값도 변한다.
피처(컬럼)의 수가 늘어나면 output layer도 같이 늘어난다.
'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 4))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3, activation='softmax')) # 이진분류모델
'''
3개 이상의 다중분류를 할때 output layer에서 사용하는 활성함수

'''
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])


print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

'''
기본값
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]

예측값
[[2.5903390e-05 9.9996805e-01 6.1043997e-06]
 [2.2149867e-05 9.9995983e-01 1.7950208e-05]
 [3.7778187e-05 9.9990463e-01 5.7594196e-05]
 [9.9999094e-01 9.1141719e-06 2.0187925e-12]
 [1.6637312e-05 9.9997783e-01 5.5199871e-06]]

원핫 인코딩 적용후
Epoch 00166: early stopping
1/1 [==============================] - 0s 13ms/step - loss: 1.4669e-04 - accuracy: 1.0000
loss :  0.00014668621588498354
accuracy :  1.0

loss :  0.0
accuracy :  0.625

'''