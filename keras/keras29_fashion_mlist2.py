# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(60000, 28, 28, 1)/255
x_test = x_test.reshape(10000, 28, 28, 1)/255
'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''
ValueError: Expected 2D array, got 1D array instead:
array=[7 2 1 ... 4 5 6].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample
'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''
특성있는 부분을 바꿔준다.
[[0. 0. 0. ... 1. 0. 0.]
 [0. 0. 1. ... 0. 0. 0.]
 [0. 1. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
(10000, 10)
[[0. 0. 0. ... 0. 0. 0.]
 [1. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 1. 0.]]
(60000, 10)
'''
# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 



# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송
model = Sequential() 
model.add(Conv2D(64, kernel_size=(2,2) ,padding = 'same' ,input_shape=(28, 28, 1)))
model.add(Conv2D(16, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(4, (2,2) ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(2, (2,2),padding = 'same' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())#(N, 180)

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=512, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''

loss :  0.9697231650352478
accuracy :  0.871999979019165

model.add(Conv2D(16, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(4, (2,2) ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(4, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(2, (2,2),padding = 'same' ,activation = 'relu'))
loss :  0.5193244218826294
accuracy :  0.8852999806404114

batch_size=256
loss :  0.6477283835411072
accuracy :  0.8632000088691711
batch_size=512
loss :  0.4586750864982605
accuracy :  0.8754000067710876
'''