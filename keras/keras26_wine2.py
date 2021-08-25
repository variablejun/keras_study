import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0) # ./ 현재폴더 ../ 상위폴더 데이터 구분자 ;
# index는 없고 헤더는 첫번째 라인


print(datasets.shape) # (4898, 12)

x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]


'''
변한것
output dimension
(4898, 1)
[[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 1. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
(4898, 7)
원핫 인코딩은 데이터의 분포를 벡터화 시킨것입니다.
원핫 인코딩을 통해서 output dimension이 바뀌습니다.
x = datasets.iloc[:,0:11]
y = datasets.iloc[:,[11]]
print(y.shape)
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
print(y)
print(y.shape)

배열 받는것
iloc 함수 행번호를 이용해서 행을 가져오는것 (마지막행 -1로도 가져옴)

OneHotEncoder
판다스를 넘파이로 바꾸고 xy분리후 y라벨확인 np.unique(y)
다중분류
모델링후 0.8이상

'''
from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
OE.fit(y)
y = OE.transform(y).toarray()
'''
데이터의 분포가 012가 아닐때 케라스를 사용하면 012부터 채워줘 라벨수가 많아진다
그래서 모델의 분석이 정확하지 않을 수 있다
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.9995, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

model = Sequential()
model.add(Dense(128,activation='relu', input_dim = 11))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy']) # 이진분류모델 에 대한 로스
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=100, mode='max', verbose=1)

model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.0003, callbacks=[es]) 

loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
MinMaxScaler
loss :  2.0201761722564697
accuracy :  0.6326530575752258

StandardScaler
loss :  3.5162062644958496
accuracy :  0.6734693646430969


RobustScaler
평균대신 중앙값을 이용하는게 데이터의 분포를 이용하는 작업에서는 훨씬더 정확할수도있고 더 잘나와서 사용
loss :  2.1332619190216064
accuracy :  0.6938775777816772

train size 0.995
loss :  4.441403865814209
accuracy :  0.7200000286102295

patience= 50
loss :  3.736879825592041
accuracy :  0.7200000286102295

patience=100
loss :  3.6057345867156982
accuracy :  0.7599999904632568

모델링 수정 후
model.add(Dense(128,activation='relu', input_dim = 11))
model.add(Dense(256,activation='relu'))
model.add(Dense(512,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))

Epoch 00375: early stopping
1/1 [==============================] - 0s 13ms/step - loss: 9.9341e-07 - accuracy: 1.0000
loss :  9.934092304320075e-07
accuracy :  1.0

Epoch 00382: early stopping
1/1 [==============================] - 0s 15ms/step - loss: 0.0014 - accuracy: 1.0000
loss :  0.0013560910010710359
accuracy :  1.0

MaxAbsScaler
loss :  1.2159374952316284
accuracy :  0.636734664440155

QuantileTransformer
loss :  2.848801851272583
accuracy :  0.6040816307067871

PowerTransformer
loss :  3.4956367015838623
accuracy :  0.6775510311126709

'''



