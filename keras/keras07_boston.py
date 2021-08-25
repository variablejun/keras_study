from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

#2.모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 13))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))

'''
loss :  47.58588790893555
r2score  0.42401892574729616
'''
#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.03) 

#4.평가/예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_pred = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

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
'''