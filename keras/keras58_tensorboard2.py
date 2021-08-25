'''
커맨드창
cd \ D드라이브로
cd study
cd _save
cd _graph
dir/w
tensorboard --logdir=.
웹주소로 ㄱㄱ
'''
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

#1.data 정제된 데이터를 얻기 위해 데이터 전처리 작업(특기가 되야한다)
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5]) 

#2.모델 구성
model = Sequential()
model.add(Dense(5, input_dim=1))  # hidden node 개수
# layer 순차적으로 아래로 내려가 전단계의 output 다음단계의 input으로 구성
model.add(Dense(1)) # 순차적 모델이기 때문에 다음 input은 당연히 5개가 된다.
model.add(Dense(2)) # 인공신경망의 레이어의 깊이(딥러닝)
model.add(Dense(3)) 
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))
# 전체 데이터를 모든 레이어를 거쳐 훈련하는 그 한번이 epochs = 1

#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs=3000, batch_size=1) 

#4.평가/예측
loss = model.evaluate(x, y) 

print('loss : ', loss)

result = model.predict([6]) 

print('6의 예측값 : ', result)
"""
5000
loss :  0.3851900100708008
6의 예측값 :  [[5.8461595]]


"""