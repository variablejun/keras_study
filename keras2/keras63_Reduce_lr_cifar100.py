#overfit 극복
# 훈련데이터를 최대한 많이
# 노멀라이제이션(정규화) 사용
# 드롭아웃
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(50000, 32* 32*3) #
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)  # train에서 사용함
x_test = scaler.transform(x_test)


x_train = x_train.reshape(50000, 32, 32, 3) # 2차원으로 리쉐입 하고 다시 4차원으로 바꿔줌
x_test = x_test.reshape(10000, 32, 32, 3)


'''
데이터 순서와 내용에 변화가 없으면 2차원배열로 바꾸어주어도 된다.
스케일러는 
  ValueError: Input 0 of layer sequential is incompatible with the layer: expected axis -1 of input shape to have value 1 but received input with shape (None, 32, 32, 3)

(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
'''
데이터 전처리와 마찬가지로 차원수를 바꾸어주는것
2차원 배열이 들어가야한다.
ValueError: Expected 2D array, got 1D array instead:
array=[7 2 1 ... 4 5 6].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample

from tensorflow.keras.utils import to_categorical
y_test = to_categorical(y_test)
y_train = to_categorical(y_train)
원핫인코딩과 차이점은 차원수에 훨씬 자유롭지만 데이터가 567이 있어도 0부터 7까지 채워서 상황에 따라 부정확하다.
'''
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)
(10000, 100)
(50000, 100)
'''
# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 



# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송
model = Sequential() 
model.add(Conv2D(128, kernel_size=(2,2) ,padding = 'valid' ,input_shape=(32,32, 3)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(32, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(16, (2,2),padding = 'valid' ,activation = 'relu'))
'''
model.add(Flatten())#(N, 180)
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
콘보루션 연산후 Fully connected에서 연산을 많이 하면 연산수가 많아져 특성값이 퍼지는데 그것을 방지하여
바로 output으로 넘겨준다.
콘보루션연산이 Fully connected에서 한 연산보다 훨씬 신뢰할수있다 가정하고 바로 넘긴다.
콘보루션 연산과 아웃풋에 비율을 맞추어 잘라서 특성값의 평균을 구한다. 
순서와 비율이 동일하기에 문제될게없다.
오버핏을 줄인다.
'''
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='sigmoid'))# 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다
from tensorflow.keras.optimizers import Adam
optimizer = Adam(lr=0.001)

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
es = EarlyStopping(monitor='accuracy', patience=15, mode='max', verbose=3)
reduce_LR = ReduceLROnPlateau(monitor='val_loss',patience=5,mode='auto',verbose=1,factor=0.5) # 러닝레이트 감소량 0.5퍼센트 50%씩 준다

import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=300, batch_size=512, validation_split=0.25, verbose=2,callbacks=[es,reduce_LR]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
GAP이후
loss :  3.5729258060455322
accuracy :  0.15360000729560852

loss :  2.713658094406128
accuracy :  0.31929999589920044

loss :  2.6359801292419434
accuracy :  0.3353999853134155

loss :  2.5316414833068848
accuracy :  0.36730000376701355

loss :  2.511610984802246
accuracy :  0.35530000925064087

loss :  2.576345205307007
accuracy :  0.3562999963760376
====================================
Reduce
0.1
Epoch 00012: ReduceLROnPlateau reducing learning rate to 0.05000000074505806.
Epoch 00018: ReduceLROnPlateau reducing learning rate to 0.02500000037252903.
loss :  4.606437683105469
accuracy :  0.009999999776482582

0.01
loss :  4.605522632598877
accuracy :  0.009999999776482582

0.001
loss :  3.341480016708374
accuracy :  0.20260000228881836
'''