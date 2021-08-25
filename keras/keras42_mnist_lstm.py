import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )

'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)

x_train = x_train.reshape(28,28,1)
x_test = x_test.reshape(28, 28,1)

'''



from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''

'''
model = Sequential()
model.add(LSTM(32,activation = 'relu',input_shape=(28,28)))
model.add(Dense(16,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(2,activation='relu'))
#model.add(GlobalAveragePooling2D())
#ValueError: Input 0 of layer global_average_pooling2d is incompatible with the layer: expected ndim=4, found ndim=2. Full shape received: (None, 8)
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다
# GAP은 4차원 입력을 받는데 2차원을 주기때문에 작동안함

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=30, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.003, verbose=1,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
배치사이즈 256 input 256
loss :  0.3374960124492645
accuracy :  0.9484000205993652

배치사이즈 128 input 512 patience=20
loss :  0.33042457699775696
accuracy :  0.9502000212669373


loss :  0.27861249446868896
accuracy :  0.9563999772071838

loss :  2.304013967514038
accuracy :  0.11349999904632568

'''