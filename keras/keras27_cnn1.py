from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten



'''
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(5,5,1)))# (N ,5 ,5 ,1) -> input shape
#(N, 4, 4, 10) -> 연산후 값
# 아웃풋을 10으로 주고 2대2로 자른다 28,28, 흑백(1)
model.add(Conv2D(20, (2,2), activation = 'relu')) # 명시하지 않아도 됨 (N, 3, 3, 20)
'''
model = Sequential() 
model.add(Conv2D(10, kernel_size=(2,2) ,input_shape=(5,5,1)))
model.add(Conv2D(20, (2,2), activation = 'relu'))
model.add(Conv2D(30, (2,2)))
model.add(Flatten())#(N, 180)
model.add(Dense(64, activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1)) 
model.summary()
'''
conv2d의 활성함수 디폴트값
콘보루션 레이어의 활성함수 default값은 linear 함수 입니다.
linear 활성함수는 입력값과 가중치에 변화를 주지 않고 그대로 보냅니다.

콘보루션의 파라미터개수의 대해 정리 및 이해
1 (input) *  2 * 2 (kernelsize) * 10(output) + 10(바이어스) = 50
10 (input) * 2 *2 (kernelsize) * 20(output) + 20(바이어스) = 820
20 (input) * 2 *2 (kernelsize) * 30(output) + 30(바이어스) = 2430

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 4, 4, 10)          50
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 3, 3, 20)          820
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 2, 2, 30)          2430
_________________________________________________________________
flatten (Flatten)            (None, 120)               0
_________________________________________________________________
dense (Dense)                (None, 64)                7744
_________________________________________________________________
dense_1 (Dense)              (None, 32)                2080
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 33
=================================================================
Total params: 13,157
Trainable params: 13,157
Non-trainable params: 0

'''