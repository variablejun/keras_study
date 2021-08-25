from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D



'''
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5,5,1)))# (N ,5 ,5 ,1) -> input shape
#(N, 4, 4, 10) -> 연산후 값
# 아웃풋을 10으로 주고 2대2로 자른다 28,28, 흑백(1)
model.add(Conv2D(20, (2,2), activation = 'relu')) # 명시하지 않아도 됨 (N, 3, 3, 20)
'''
model = Sequential() 
model.add(Conv2D(10, kernel_size=(2,2) ,padding = 'same' ,input_shape=(10,10,1))) # N 10 10 1
# padding = 'same' -> 그 다음 레이어에 동일한 쉐입으로 전달해준다
# N 10 10 10 출력된 값
model.add(Conv2D(20, (2,2), activation = 'relu')) # N 9 9 20 -> 커널 사이즈에 따라서 줄어드는 수가 다르다.
model.add(Conv2D(30, (2,2), padding='valid'))# N 8 8 30
model.add(MaxPooling2D())# N 4 4 30
# 조각 조각나눈것중에 최대값을 가져와 다시 맞추는데 그럴때 쉐입 크기가 반으로 줄어든다
# 맥스풀링하고 또 콘보루션 연산이 가능하다
model.add(Conv2D(15, (3,3)))
model.add(Flatten())#(N, 480)
# 데이터의 내용과 순서는 변하지 않고 차원수만 낮추어 댄스연산이 가능하게 만든다
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1)) 
model.summary()
'''
conv2d의 활성함수 디폴트값
콘보루션의 파라미터개수의 대해 정리 및 이해
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 10, 10, 10)        50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 20)          820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 8, 8, 30)          2430
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 4, 4, 30)          0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 2, 2, 15)          4065
_________________________________________________________________
flatten (Flatten)            (None, 60)                0
_________________________________________________________________
dense (Dense)                (None, 64)                3904
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 13,382
Trainable params: 13,382
Non-trainable params: 0
'''