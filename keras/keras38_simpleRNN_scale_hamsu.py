import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],
[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) # 묶음단위 timesteps
# 각 timesteps에 feature단위로 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predic = np.array([50,60,70]).reshape(1,3,1)
print(x.shape, y.shape) # (13, 3) (13,)
x = x.reshape(x.shape[0],x.shape[1],1)

#inputs: A 3D tensor, with shape [batch, timesteps, feature].(4,3,1) 

#model.add(SimpleRNN(16,activation = 'relu',input_shape=(3,1))) # units RNN, DNN fliter CNN


input1 = Input(shape=(3,1))
xx = SimpleRNN(32,activation = 'relu')(input1)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output1 = Dense(2)(xx)
model = Model(inputs = input1, outputs = output1)

'''
input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)

model = Model(inputs = input1, outputs = output1)
'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es]) 
#inputs = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_predic)
print(results[0])

'''
스케일러적용시
[1163.8608 1197.1864]
[519.4245  515.76184]

스케일러미적용
[79.80634 79.9166 ]
'''