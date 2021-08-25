from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x = datasets.data
y = datasets.target
'''
데이터 전처리 과정 
minmax scalery : 모든 데이터를 최대값으로 나누어 최소값은  최대값은1로 바꾸어 연산해서 exploding을 방지하고 성능을향상시키는것
데이터의 비율은 변하지 않기 때문에 y값도 변하지 않는다. 
데이터의 간격만 변하지 않으면 y는 왜곡되지 않는다.
데이터의 값은 바뀌였지만 동일한 비율로 압축되었기 때문에 데이터가 가리키는 y의 값은 바뀌지 않는다.
공식 (개체 - min)/ (max - min)
numpy는 부동소수점 연산에 특화되어있다
'''
print(np.max(x))

# x = (x - np.min(x)) / (np.max(x) - np.min(x))
# x열마다 전처리 과정을 해주기 for, append 해주는 라이브러리있음
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x) # 사이킷런의 실행개념
x_scale = scaler.transform(x) # fit 한걸 저장
# 열마다 MinMaxScaler로 전처리를 해서 모든 데이터 전처리했을때보다 성능을더 좋게 만든다
#print(x_scale[:])
#print(np.max(x_scale), np.min(x_scale)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale,y,
train_size = 0.95, random_state=66)

#2.모델 구성
model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

'''
열 별 데이터 전처리
loss :  8.539509773254395
r2score  0.933771570286663 

전처리후 수치
loss :  9.419081687927246
r2score  0.926950027562462

47/47
train_size = 0.95 와 batch_size=10에 따라서 달라진다.
loss :  10.399333953857422
r2score  0.9193476432016088
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