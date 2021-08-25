
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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

#scaler = MinMaxScaler()
scaler = PowerTransformer()
scaler.fit(x_train) # x train를 스케일 하여 스케일링 기준을 xtrain으로 만들고 
x_train = scaler.transform(x_train) #
x_test = scaler.transform(x_test) # xtrain 기준의 scaler를 test에 적용시킨다


'''
전체 데이터를 전처리 했을 경우에 test데이터도 전처리 되어 train에 과적합 된다. test는 train의 범위밖에(예외)
두어야한다. 과적합을 방지하고 제대로된 평가를 하기 위해서 따로 해야하고 fit은 train만 해야한다.
예외를 인정해야하고 testset 도 train의 비율대로 스케일 해야한다.

'''
# 열마다 MinMaxScaler로 전처리를 해서 모든 데이터 전처리했을때보다 성능을더 좋게 만든다
#print(x_scale[:])
#print(np.max(x_scale), np.min(x_scale)



#2.모델 구성
model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.03) 

#4.평가/예측
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss)
y_pred = model.predict(x_test) 

# y_pred = scaler.transform(y_pred)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)


'''
StandardScaler
1.
loss :  5.204648971557617
r2score  0.9596351835506617
2.
loss :  8.25940990447998
r2score  0.9359439001222207

MinMaxScaler
1.
loss :  7.7063679695129395
r2score  0.940233027543668
2.
loss :  6.391960144042969
r2score  0.9504269546745601

RobustScaler
1.
loss :  7.733130931854248
r2score  0.9400254709065787
2.
loss :  11.545578956604004
r2score  0.9104579100639526

MaxAbsScaler
1.
loss :  8.78338623046875
r2score  0.9318801768588283
2.
loss :  8.088309288024902
r2score  0.937270863890056

QuantileTransformer
1.
loss :  8.70199203491211
r2score  0.9325114379884315
2.
loss :  8.412805557250977
r2score  0.9347542333898381

PowerTransformer
1.
loss :  6.6814165115356445
r2score  0.9481820735460025
2.
loss :  6.860228538513184
r2score  0.946795297742345
3. 
'''