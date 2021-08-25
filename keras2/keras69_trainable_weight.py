import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

x = np.array([1,2,3,4,5])
y = np.array([1,2,3,4,5])

model = Sequential()
model.add(Dense(3,input_dim=1))

model.add(Dense(2))
model.add(Dense(1))
model.summary()
print(model.weights)
print('===============================')
print(model.trainable_weights)
print('===============================')
print(len(model.weights))
print(len(model.trainable_weights))
'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 3)                 6
_________________________________________________________________
dense_1 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 3
=================================================================
Total params: 17
Trainable params: 17
Non-trainable params: 0

[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.5869812 , -0.02350819, -0.9152941 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.26385778,  0.71791875],
       [ 1.092597  , -0.29817688],
       [ 0.3961432 ,  0.34152246]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 
1) dtype=float32, numpy=
array([[ 0.75033057],
       [-0.02523994]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
===============================
[<tf.Variable 'dense/kernel:0' shape=(1, 3) dtype=float32, numpy=array([[ 0.5869812 , -0.02350819, -0.9152941 ]], dtype=float32)>, <tf.Variable 'dense/bias:0' shape=(3,) dtype=float32, numpy=array([0., 0., 0.], dtype=float32)>, <tf.Variable 'dense_1/kernel:0' shape=(3, 2) dtype=float32, numpy=
array([[-0.26385778,  0.71791875],
       [ 1.092597  , -0.29817688],
       [ 0.3961432 ,  0.34152246]], dtype=float32)>, <tf.Variable 'dense_1/bias:0' shape=(2,) dtype=float32, numpy=array([0., 0.], dtype=float32)>, <tf.Variable 'dense_2/kernel:0' shape=(2, 
1) dtype=float32, numpy=
array([[ 0.75033057],
       [-0.02523994]], dtype=float32)>, <tf.Variable 'dense_2/bias:0' shape=(1,) dtype=float32, numpy=array([0.], dtype=float32)>]
6    3(layer)(w+b)  
6    3(layer)(w+b)
'''