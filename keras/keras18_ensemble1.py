# 데이터 구성

import numpy as np 
x1 = np.array([range(100), range(301,401), range(1,101)])
x2 = np.array([range(101,201), range(411,511), range(100,200)])
x1 = np.transpose(x1)
x2 = np.transpose(x2)
y = np.array(range(1001,1101))
print(x1.shape, x2.shape, y.shape) # (100, 3) (100, 3) (100,)

from sklearn.model_selection import train_test_split

x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1,x2,y,train_size = 0.7, random_state=66) # train_size 0.7
print(x1_train.shape, 
x1_test.shape, 
x2_train.shape,
x2_test.shape, 
y_train.shape,
y_test.shape )
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
output2 = Dense(1)(dence14)
# output이 나왔지만 히든레이어

from tensorflow.keras.layers import concatenate, Concatenate  #  소문자 메소드,  대문자 클래스
merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
last_output1 = Dense(1)(merge3)

model = Model(inputs = [input1, input2], outputs = last_output1)

model.compile(loss = 'mse', optimizer = 'adam', metrics=['mae'])
model.fit([x1_train,x2_train],y_train, epochs=100, batch_size=10, validation_split=0.3, shuffle=0.2) 
# model.fit([x1,x2],y, epochs=100, batch_size=10, validation_split=0.3, shuffle=0.2) 

loss = model.evaluate([x1_test,x2_test], y_test) 
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