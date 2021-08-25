from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=66)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


#2.모델 구성
input1 = Input(shape=(10,))
dense1 = Dense(400)(input1)
dense2 = Dense(1000)(dense1)
dense3 = Dense(100)(dense2)
dense4 = Dense(50)(dense3)
dense5 = Dense(20)(dense4)
dense6 = Dense(10)(dense5)
dense7 = Dense(5)(dense6)
dense8 = Dense(2)(dense7)

output1 = Dense(1)(dense8)

model = Model(inputs = input1, outputs = output1) # 모델 여러개를 합치거나 순서를 바꾸기 쉽다.
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 10)]              0
_________________________________________________________________
dense (Dense)                (None, 400)               4400
_________________________________________________________________
dense_1 (Dense)              (None, 1000)              401000
_________________________________________________________________
dense_2 (Dense)              (None, 100)               100100
_________________________________________________________________
dense_3 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_4 (Dense)              (None, 20)                1020
_________________________________________________________________
dense_5 (Dense)              (None, 10)                210
_________________________________________________________________
dense_6 (Dense)              (None, 5)                 55
_________________________________________________________________
dense_7 (Dense)              (None, 2)                 12
_________________________________________________________________
dense_8 (Dense)              (None, 1)                 3
=================================================================
Total params: 511,850
Trainable params: 511,850
Non-trainable params: 0
_________________________________________________________________
5/5 [==============================] - 0s 1ms/step - loss: 3614.9741
loss :  3614.97412109375
r2score  0.41978181546019233

'''
'''
models = Sequential()
models.add(Dense(400, input_shape=(10,)))
models.add(Dense(1000))
models.add(Dense(100))
models.add(Dense(50))
models.add(Dense(20))
models.add(Dense(10))
models.add(Dense(5))
models.add(Dense(2))
models.add(Dense(1))
Sequential
loss :  3074.521728515625
r2score  0.5065266670105661
'''
#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3, shuffle=0.2) 
model.summary()
#4.평가/예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)


y_pred = model.predict(x_test) 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
'''
r2score  0.5037277375596032
'''