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
model.add(Dense(8,activation='relu' , input_shape=(28,28)))

model.add(Flatten())#(N, 180) 다차원 데이터를 DENSE로 구현할 때 reshape 하지 않고 펴버릴수잇다
#전처리 과정에서 말고 모델링에서도 가능하다
model.add(Dense(10,activation='relu'))
model.summary()