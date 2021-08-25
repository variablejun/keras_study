from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = PowerTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


#2.모델 구성
model = Sequential()
model.add(Dense(40,activation='relu', input_shape=(10,)))
model.add(Dense(500,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(1))


#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.03) 

#4.평가/예측
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss)
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
StandardScaler
평균과 표준편차를 이용한 스케일러 방법으로 아웃라이어의 영향을 가장 많이 받는 방식입니다

1.
loss :  9107.7177734375
r2score  -0.22118241719039355
2.
loss :  11235.546875
r2score  -0.5064865806800729

MinMaxScaler
(x - min)/(max- min) 수식을 이용하여 모든 값들을 0과 1사이에 넣어서 좁은값에 동일한 비율로 압축시켜 스케일하는 방법입니다.
1.
loss :  5540.55419921875
r2score  0.2571104935655508
2.
loss :  5054.81787109375
r2score  0.3222391044020948

RobustScaler
평균값이 아닌 중위값으로 스케일하여 아웃라이어의 영향을 최대한 줄이고 값이 더 넓게 분포하도록 하는 방식입니다.
아웃라이어의 영향이 클 때 사용합니다.
1.
loss :  10184.5029296875
r2score  -0.36556031498808017
2.
loss :  10381.197265625
r2score  -0.3919335595718716

MaxAbsScaler
절대값을 이용하기 때문에 음수데이터손실이 적어 음수데이터를 스케일 할때 이용합니다.
데이터가 양수만 존재할 경우 min max와 비슷한 결과가나올 수 있습니다.
1.
loss :  8384.34765625
r2score  -0.12419151413565244
2.
loss :  10531.591796875
r2score  -0.41209856014463164

QuantileTransformer
1000개의 분위를 사용하여 데이터를 분포시켜 아웃라이어의 영향을 최소화 하고 0과 1사이로 압축시켜 스케일합니다
비선형으로 변환되어 선형상관관계를 왜곡시킬수있으나 다른척도에서 측정된 변수를 직접적으로 비교할수있습니다.
keras23_EarlyStopping.py
1.
loss :  6717.26904296875
r2score  0.0993340965486087
2.
loss :  8121.9775390625
r2score  -0.08901234379639411

PowerTransformer
데이터를 정규분포와 유사하게 만들기 위해 거듭제곱 변환을 하여 스케일하는 방법입니다.
이분산성이나 정규성이 필요한 모델에 적합합니다.
동분산성 평균제곱의 오차가 상수인 경우를 말합니다.
이런 동분산이 결여되어 분산이 독립변수에 띠라 달라지는 경우를 이분산성이라고 합니다.

1.
loss :  9117.822265625
r2score  -0.22253745775627753
2.
loss :  9959.384765625
r2score  -0.33537585660348634
'''