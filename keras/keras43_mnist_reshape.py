import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Reshape

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(60000, 28, 28)
x_test = x_test.reshape(10000, 28, 28)

from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''


'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''

'''
# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 



# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송
model = Sequential() 
# model.add(Conv2D(50, kernel_size=(2,2) ,padding = 'same' ,input_shape=(28, 28, 1)))
model.add(Dense(10,activation='relu' , input_shape=(28,28)))

model.add(Flatten())#(N, 180) 다차원 데이터를 DENSE로 구현할 때 reshape 하지 않고 펴버릴수잇다
#전처리 과정에서 말고 모델링에서도 가능하다

model.add(Reshape((28,10,1))) # input 쉐입이랑 크기가 같아야한다.
model.add(Conv2D(22,(2,2)))
model.add(Conv2D(22,(2,2)))
model.add(Conv2D(22,(2,2)))
model.add(Conv2D(22,(2,2)))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 28, 8)             232
_________________________________________________________________
flatten (Flatten)            (None, 224)               0
_________________________________________________________________
dense_1 (Dense)              (None, 10)                2250
=================================================================
Total params: 2,482
Trainable params: 2,482
Non-trainable params: 0
'''
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=50, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.003, verbose=1,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
Epoch 00237: early stopping
313/313 [==============================] - 1s 4ms/step - loss: 0.1640 - accuracy: 0.9778
loss :  0.16404393315315247
accuracy :  0.9778000116348267
'''