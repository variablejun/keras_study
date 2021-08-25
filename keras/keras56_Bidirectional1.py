import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, Bidirectional
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6]]) # 묶음단위 timesteps
# 각 timesteps에 feature단위로 
y = np.array([4,5,6,7])

print(x.shape, y.shape) # (4, 3) (4,)

x = x.reshape(4,3,1)


#inputs: A 3D tensor, with shape [batch, timesteps, feature].(4,3,1) 
model = Sequential()
#model.add(SimpleRNN(16,activation = 'relu',input_shape=(3,1))) # units RNN, DNN fliter CNN
model.add(LSTM(16,activation = 'relu',input_shape=(3,1), return_sequences=True))
model.add(Bidirectional(LSTM(9,activation = 'relu'))) 

# 앞에와 뒤에 inputshape 맞추어야 하는데 중간에있을땐 In out 노드수가 명시되어있는데 input레이어에선 아니기때문에
# input에서 쓰기 어렵다
# 패딩되어서 0으로 가서 연산 할 수있기때문에 판단은 자유 무조간 좋지않다.
# 또한 in put 바로 아래쓸때는 차원수가 줄기에 리턴시퀀스 아용해야함
model.add(Dense(9 ,activation='relu'))
model.add(Dense(1 ,activation='relu'))
model.summary()
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lstm (LSTM)                  (None, 3, 16)             1152
_________________________________________________________________
bidirectional (Bidirectional (None, 18)                1872
_________________________________________________________________
dense (Dense)                (None, 9)                 171
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 10
=================================================================
Total params: 3,205
Trainable params: 3,205
Non-trainable params: 0
_________________________________________________________________
PS D:\study> 
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
'''
[[8.067264]]
'''