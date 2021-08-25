import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
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
model.add(GRU(32,activation = 'relu',input_shape=(3,1)))

model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1 ,activation='relu'))
'''

'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x, y, epochs=10000, batch_size=1, callbacks=[es]) 
#inputs = np.array([5,6,7]).reshape(1,3,1)

results = model.predict(x_predic)
print(results)

'''


'''