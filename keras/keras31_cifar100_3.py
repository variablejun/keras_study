#모델링완성 하시오 32,32,3
#모델링완성 하시오 32,32,3
# 모델링 완성
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D

(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(50000, 32* 32*3) #
x_test = x_test.reshape(10000, 32*32*3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
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
model.add(Conv2D(64, kernel_size=(2,2) ,padding = 'valid' ,input_shape=(32,32, 3)))
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())#(N, 180)

model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(100, activation='sigmoid'))# 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=10, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=100, batch_size=64, validation_split=0.25, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))
#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc= 'upper right')
'''
#2
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker=',', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker=',', c='blue', label='val_loss')
plt.grid()
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.legend(loc= 'upper right')

'''
'''

StandardScaler
걸린시간 170.49719905853271
loss :  15.498335838317871
accuracy :  0.1412999927997589

걸린시간 355.2093777656555
loss :  12.09375286102295
accuracy :  0.1574999988079071

걸린시간 313.04285526275635
loss :  5.612170219421387
accuracy :  0.19020000100135803

model.add(MaxPooling2D())
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Conv2D(8, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(64, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(MaxPooling2D())

걸린시간 319.4718985557556
loss :  7.428462505340576
accuracy :  0.24199999868869781

걸린시간 311.248247385025
loss :  6.669098377227783
accuracy :  0.26930001378059387

MinMaxScaler
걸린시간 550.3497042655945
loss :  23.22853660583496
accuracy :  0.1429000049829483

RobustScaler
걸린시간 844.5119268894196
loss :  21.482006072998047
accuracy :  0.16609999537467957

loss :  18.9493350982666
accuracy :  0.1445000022649765

loss :  3.57077693939209
accuracy :  0.15070000290870667

'''