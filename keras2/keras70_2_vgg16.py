from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19

vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))
vgg16.trainable=False 
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
# vgg16을 연산하지않아 가중치에 변화없이 기존에 모델링으로 주는것, 가중치 동결 훈련 동결

model.summary()
print(len(model.weights))
print(len(model.trainable_weights))
'''
Full Con
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 3, 3, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
dense (Dense)                (None, 10)                46090
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 14,760,789
Trainable params: 0
Non-trainable params: 14,760,789
_________________________________________________________________
30
0



_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
vgg16 (Functional)           (None, 3, 3, 512)         14714688
_________________________________________________________________
flatten (Flatten)            (None, 4608)              0
_________________________________________________________________
dense (Dense)                (None, 10)                46090
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 11
=================================================================
Total params: 14,760,789
Trainable params: 46,101  # 트레이닝한 개수
Non-trainable params: 14,714,688 # 동결한것
_________________________________________________________________
30
4
'''