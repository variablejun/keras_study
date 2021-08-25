# 데이터 구성

import numpy as np 
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y1 = np.array(range(1001,1101))
y2 = np.array(range(1101,1201))

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2,train_size = 0.7, random_state=66) # train_size 0.7

print(x1_train.shape, 
x1_test.shape, 
x2_train.shape,
x2_test.shape, 
y1_train.shape,
y1_test.shape,
y2_train.shape,
y2_test.shape )
#(70, 3) (30, 3) (70, 3) (30, 3) (70,) (30,) (70,) (30,)

# 여러개의 리스트를 train과 test로 나눌때

# 모델 구성
# 함수형 모델은 앙상블 형태의 모델을 짤때 편리하게 사용할 수 있다

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
# 연산은 같이 하나 서로 영향을 주지않는다.
# 2-1

input1 = Input(shape=(3,)) 
dence1 = Dense(10, activation='relu')(input1)
dence2 = Dense(7, activation='relu')(dence1)
dence3 = Dense(5, activation='relu')(dence2)
output1 = Dense(1)(dence3)


# 2-2

input2 = Input(shape=(3,)) 
dence11 = Dense(10, activation='relu')(input2)
dence12 = Dense(7, activation='relu')(dence11)
dence13 = Dense(5, activation='relu')(dence12)
dence14 = Dense(5, activation='relu')(dence13)
output2 = Dense(1)(dence14) # 실제 아웃풋이 아니라 활성화함수써됨 마지막 레이어에서는 따로 사용하는 default 활성화함수 사용
# output이 나왔지만 히든레이어
# dense_14 (Dense)                (None, 1)            8           dense_13[0][0]
# _14는 dense명이아닌 레이어의 깊이를뜻한다
from tensorflow.keras.layers import concatenate  #  소문자 메소드,  대문자 클래스
from tensorflow.keras.layers import Concatenate

merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)

output21 = Dense(7)(merge3)# 원하는 레이어를 붙여주기만 하면 거기에서 바로 분기시킬 수 있다
last_output1 = Dense(1)(output21) 

output22 = Dense(7)(merge3)
last_output2 = Dense(1)(output22)

model = Model(inputs = [input1, input2], outputs = [last_output1,last_output2])
model.summary()
model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train,x2_train],[y1_train,y2_train ], epochs=10, batch_size=10, validation_split=0.3, shuffle=0.2) 
# model.fit([x1,x2],y, epochs=100, batch_size=10, validation_split=0.3, shuffle=0.2) 

loss = model.evaluate([x1_test,x2_test], [y1_test, y2_test]) 
print(loss)
'''
loss: 1720760.7500 - dense_12_loss: 653594.3750 - dense_14_loss: 1067166.3750 - dense_12_mae: 808.3802 - dense_14_mae: 1032.9694
[1720760.75, 653594.375, 1067166.375, 808.3802490234375, 1032.9693603515625]
'''
print('loss : ', loss[0])
print(' metrics=[mae] : ', loss[1])
# loss = model.evaluate([x1_test,x2_test],y_test) 

'''
y_pred = model.predict([x1_test,x2_test], y_test) 

print(y_pred) # mes, mae

from sklearn.metrics import r2_score
r2 = r2_score([y1_test,y2_test], y_pred)

print("r2score ", r2)

'''