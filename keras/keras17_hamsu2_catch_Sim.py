import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

x = np.array([1,2,3,4,5]) #numpy 형식로바꿔줌 완성후 출력스샷(loss, predic)
y = np.array([1,2,4,3,5]) # 판단은 loss값으로

x_pred = np.array([6])


input1 = Input(shape=(1,))
dense1 = Dense(2)(input1)
dense2 = Dense(3)(dense1)
dense3 = Dense(2)(dense2)
output1 = Dense(1)(dense3)

model = Model(inputs = input1, outputs = output1) # 모델 여러개를 합치거나 순서를 바꾸기 쉽다.
model.summary()
'''
Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
input_1 (InputLayer)         [(None, 1)]               0
_________________________________________________________________
dense (Dense)                (None, 2)                 4
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 9
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 8
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 3
=================================================================
Total params: 24
Trainable params: 24
Non-trainable params: 0

Loss :  0.3800446689128876
예측값 :  [[1.1974976]
 [2.095674 ]
 [2.99385  ]
 [3.892027 ]
 [4.790203 ]]
r2 score :  0.8099776602785482
'''
'''
model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(2))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))


Sequential
Loss :  0.3800089359283447
예측값 :  [[1.2022027]
 [2.1025746]
 [3.0029464]
 [3.9033182]
 [4.80369  ]]
r2 score :  0.8099955260250098
'''
model.compile(loss="mse", optimizer="adam")
model.fit(x, y, epochs=1000, batch_size=1)

model.summary()

loss = model.evaluate(x,y)
print('Loss : ', loss)

x_pred = model.predict(x)
print('예측값 : ', x_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y, x_pred) # y의원래값과 y의 예측값

print('r2 score : ', r2)

# 리더보드 방식 -> 최고점을 항상 갱신시키는 방식
# 0.9까지