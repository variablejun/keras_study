#overfit 극복
# 훈련데이터를 최대한 많이
# 노멀라이제이션(정규화) 사용
# 드롭아웃
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100, mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout

(x_train, y_train), (x_test, y_test) = mnist.load_data() 
print(x_train.shape,y_train.shape ) # (60000, 28, 28)
print(x_test.shape,y_test.shape )


x_train = x_train.reshape(60000, 28*28*1) #
x_test = x_test.reshape(10000, 28*28*1)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # train에서 사용함
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)



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
# Conv2D에 들어가 연산하기 위해 3차원 데이터를 4차원 데이터로 쉐입을 바꾸어준다. 



# 0.992 이상으로만들기 , 캡쳐후 단톡에 전송


model = Sequential() 
model.add(Conv2D(128, kernel_size=(2,2) ,padding = 'valid' ,input_shape=(28,28,1)))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(16, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(Dropout(0.2))
model.add(MaxPooling2D())
model.add(Conv2D(64, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(16, (2,2),padding = 'valid' ,activation = 'relu'))
model.add(Flatten())#(N, 180)
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(10, activation='softmax'))# 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다
'''
model = load_model("./_save/keras45_1_save_model.h5") # 모델불러오기


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=20, mode='max', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.25, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
저장한 모델링값
Epoch 00151: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1048 - accuracy: 0.9909 
걸린시간 707.1952078342438
loss :  0.10479500889778137
accuracy :  0.9908999800682068

모델링 가져오고난 후 
모델만 가져오고 가중치는 저정하지 않앗기 때문에 새로 훈련시킨것이고 다르게나온다.
Epoch 00088: early stopping
313/313 [==============================] - 1s 3ms/step - loss: 0.1081 - accuracy: 0.9883
걸린시간 407.09918308258057
loss :  0.10809940099716187
accuracy :  0.9883000254631042


'''