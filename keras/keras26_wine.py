import numpy as np 
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
# acc 0.8이상

dataset = load_wine()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (178, 13) (178,)
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
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
model.add(Dense(50,activation='relu', input_dim = 13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])


print(y_test[:5])
y_predict = model.predict(x_test[:5])
print(y_predict)

'''
EarlyStopping 아큐러시로 잡았을때
es = EarlyStopping(monitor='accuracy', patience=5, mode='max', verbose=1)
Epoch 19/100
12/12 [==============================] - 0s 5ms/step - loss: 0.0358 - accuracy: 0.9987 - val_loss: 0.0916 - val_accuracy: 0.9608
Epoch 00019: early stopping
1/1 [==============================] - 0s 13ms/step - loss: 0.0126 - accuracy: 1.0000
loss :  0.012608055956661701
accuracy :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[1.6300848e-03 4.4186100e-02 9.5418382e-01]
 [2.5643059e-03 9.9658489e-01 8.5070293e-04]
 [1.4673298e-03 9.9834585e-01 1.8681282e-04]
 [9.8844248e-01 7.6929699e-03 3.8645410e-03]
 [2.1421827e-02 9.7783798e-01 7.4017298e-04]]

loss :  1.854365621056786e-07
accuracy :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[3.9584133e-08 1.4119719e-06 9.9999857e-01]
 [1.3178764e-12 1.0000000e+00 1.9959149e-12]
 [3.9484231e-14 1.0000000e+00 4.9019850e-14]
 [9.9999976e-01 2.9622698e-16 1.9680739e-07]
 [1.1991074e-08 1.0000000e+00 1.4482911e-11]]


loss :  1.3245475827261544e-08
accuracy :  1.0
[[0. 0. 1.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]]
[[4.9910062e-11 4.4452847e-08 1.0000000e+00]
 [3.1884090e-17 1.0000000e+00 3.9548053e-15]
 [3.4124013e-24 1.0000000e+00 5.5052840e-21]
 [1.0000000e+00 5.8398260e-09 1.6114132e-08]
 [6.5636704e-11 1.0000000e+00 9.0516566e-11]]
'''