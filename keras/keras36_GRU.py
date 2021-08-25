import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # 묶음단위 timesteps
# 각 timesteps에 feature단위로 
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1)


#inputs: A 3D tensor, with shape [batch, timesteps, feature].(4,3,1) 
model = Sequential()
#model.add(SimpleRNN(16,activation = 'relu',input_shape=(3,1))) # units RNN, DNN fliter CNN
model.add(GRU(16,activation = 'relu',input_shape=(3,1)))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))
model.summary()
'''
GRU 조경현 교수제안
3 * units(output) * (units + input_dim + b + reset)
tensor가 1일때 reset은 안붙음
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
gru (GRU)                    (None, 16)                912
_________________________________________________________________
dense (Dense)                (None, 32)                544
_________________________________________________________________
dense_1 (Dense)              (None, 16)                528
_________________________________________________________________
dense_2 (Dense)              (None, 8)                 136
_________________________________________________________________
dense_3 (Dense)              (None, 4)                 36
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_5 (Dense)              (None, 1)                 3
=================================================================
Total params: 2,169
Trainable params: 2,169
Non-trainable params: 0

'''
'''
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es]) 
inputs = np.array([5,6,7]).reshape(1,3,1)
results = model.predict(inputs)
print(results)



'''