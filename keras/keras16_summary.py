import numpy as np

#  데이터구성
x = np.array([range(100),range(301,401),range(1,101),range(100),range(401,501)])
x = np.transpose(x)
y = np.array([range(711,811),range(101,201)])
y = np.transpose(y)
print(x.shape , y.shape) # 100,5  100,2

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(3, input_shape=(5,)))  # x 의행(열)
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(3))
model.add(Dense(2)) # y 의행(열)

model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 18
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_2 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_3 (Dense)              (None, 3)                 12
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 8
=================================================================
Total params: 62
Trainable params: 62
Non-trainable params: 0
_________________________________________________________________
'''