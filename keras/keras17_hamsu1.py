import numpy as np

#  데이터구성
x = np.array([range(100),range(301,401),range(1,101),range(100),range(401,501)])
x = np.transpose(x)
y = np.array([range(711,811),range(101,201)])
y = np.transpose(y)
print(x.shape , y.shape) # 100,5  100,2

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

input1 = Input(shape=(5,))
dense1 = Dense(3)(input1)
dense2 = Dense(4)(dense1)
dense3 = Dense(10)(dense2)
output1 = Dense(2)(dense3)

model = Model(inputs = input1, outputs = output1) # 모델 여러개를 합치거나 순서를 바꾸기 쉽다.
# 모델을 명시하는 위치가 시퀀셜형과 다르게 명시해준다
# (inputshape + b) * outputshape
model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 5)]               0
_________________________________________________________________
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_2 (Dense)              (None, 10)                50
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 22
=================================================================
Total params: 106
Trainable params: 106
Non-trainable params: 0
_________________________________________________________________
'''
'''
model = Sequential()
model.add(Dense(3, input_shape=(5,)))  # x 의행(열)
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(2)) # y 의행(열)
'''
# model.summary()
'''

'''