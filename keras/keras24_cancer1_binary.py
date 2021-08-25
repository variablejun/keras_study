from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
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

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 30))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1, activation='sigmoid')) # 이진분류모델
'''
output layer의 반환값을 0~1사이에 값으로 만들주늗것
'''
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1]) # R2는 회귀모델 accuracy는 분류모델

print(y_test[-5:-1])
y_predict = model.predict(x_test[-5:-1])
print(y_predict)


'''

[1 1 0 1]

[[9.9513751e-01]
 [9.9998927e-01]
 [3.5066319e-06]
 [9.9999845e-01]]

시그모이드, 바이너리크로스엔트로피 분류모델
activation='sigmoid'
loss = 'binary_crossentropy'
train size 0.7
loss :  0.2837662398815155
accuracy :  0.9590643048286438

train size 0.95
Epoch 00075: early stopping
1/1 [==============================] - 0s 12ms/step - loss: 7.3777e-04 - accuracy: 1.0000
loss :  0.0007377693546004593
accuracy :  1.0


회귀모델
38/38 [==============================] - 0s 3ms/step - loss: 0.0064 - val_loss: 0.0260
Epoch 00189: early stopping
1/1 [==============================] - 0s 11ms/step - loss: 5.8222e-04
loss :  0.0005822232342325151

38/38 [==============================] - 0s 3ms/step - loss: 2.9602e-04 - val_loss: 0.0270
Epoch 00065: early stopping
1/1 [==============================] - 0s 12ms/step - loss: 0.0053
loss :  0.005255149211734533

38/38 [==============================] - 0s 3ms/step - loss: 5.3393e-04 - val_loss: 0.0286
Epoch 00066: early stopping
1/1 [==============================] - 0s 13ms/step - loss: 0.0130
loss :  0.013011676259338856

'''