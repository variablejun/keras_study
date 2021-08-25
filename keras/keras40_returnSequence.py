import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],
[10,11,12],[20,30,40],[30,40,50],[40,50,60]]) # 묶음단위 timesteps
# 각 timesteps에 feature단위로 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x_predic = np.array([50,60,70]).reshape(1,3,1)
print(x.shape, y.shape) # (13, 3) (13,)

x = x.reshape(x.shape[0],x.shape[1],1)

#inputs: A 3D tensor, with shape [batch, timesteps, feature].(4,3,1) 
model = Sequential()
#model.add(SimpleRNN(16,activation = 'relu',input_shape=(3,1))) # units RNN, DNN fliter CNN
model.add(LSTM(32,activation = 'relu',input_shape=(3,1),return_sequences=True))
model.add(LSTM(16,activation = 'relu',))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))
'''
return_sequences=True lstm의 input 3 output 2차원 인데 
lstm끼리 엮으려면 차원수를 맞추어야 한다 그때 사용하여 차원수를 맞춰준다
기본값은 False
여러개을 잘연결하지 않는 이유는 연속적인 자료에 많이 쓰이는데
lstm을 통과한 데이터가 연속적이라는 보장이 없고 대부분 DENSE형태이기 때문에
잘 사용하지 않는다.
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 32)             4352
_________________________________________________________________
lstm_1 (LSTM)                (None, 32)                8320
_________________________________________________________________
dense (Dense)                (None, 8)                 264
_________________________________________________________________
dense_1 (Dense)              (None, 4)                 36
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 12,985
Trainable params: 12,985
Non-trainable params: 0
'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es]) 
#inputs = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_predic)
print(results)

'''
[[83.32187]]
'''