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
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # train에서 사용함
x_test = scaler.transform(x_test)



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

model.add(Dense(1024,activation='relu',input_shape=(32* 32*3,)))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(100, activation='sigmoid'))# 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

'''
콘보루션 연산후 Fully connected에서 연산을 많이 하면 연산수가 많아져 특성값이 퍼지는데 그것을 방지하여
바로 output으로 넘겨준다.
콘보루션연산이 Fully connected에서 한 연산보다 훨씬 신뢰할수있다 가정하고 바로 넘긴다.
콘보루션 연산과 아웃풋에 비율을 맞추어 잘라서 특성값의 평균을 구한다. 
순서와 비율이 동일하기에 문제될게없다.
오버핏을 줄인다.
'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])

'''
StandardScaler
걸린시간 60.21109747886658
loss :  4.368515491485596
accuracy :  0.20600000023841858

걸린시간 84.6844265460968
loss :  4.014658451080322
accuracy :  0.20960000157356262

걸린시간 78.38208365440369
loss :  0.04740132763981819
accuracy :  0.2329999953508377

걸린시간 179.6003556251526
loss :  0.04681005701422691
accuracy :  0.2378000020980835

'''