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
train_size = 0.7, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scaler = MinMaxScaler()
scaler = StandardScaler()
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
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
'''
EarlyStopping : loss 와 valoss의 지표를 확인하여 자기가 원하는 지점에서 멈추는것
페이션스이후 20번째까지 fit 한다 그래서 최소값이 없으면 멈춘다.
'''
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, callbacks=[es]) 
print(hist.history.keys())
print(hist.history['loss'])
print(hist.history['val_loss'])

'''
모델에 fit에서 나오는 loss와 val_loss를 hist에 담아 출력
hist로 차트를 만들거나 다양한 방식으로 활용가능

dict_keys(['loss', 'val_loss'])
[580.9186401367188, 511.85400390625, 278.9764404296875, 68.00946807861328, 36.6590690612793, 29.541542053222656, 25.69819450378418, 23.553691864013672, 22.509960174560547, 20.26580047607422]
[610.3589477539062, 461.07257080078125, 104.98711395263672, 55.934478759765625, 46.4981575012207, 41.622169494628906, 37.41883850097656, 34.60236740112305, 33.30554962158203, 31.33160400390625]
'''
#4.평가/예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
'''
5/5 [==============================] - 0s 2ms/step - loss: 13.0044
loss :  13.004377365112305

evalu 값이 나온다 fit출력이 되듯 evaluate도 출력한다.
테스트 데이터를 배치사이즈로 나줘준것 evaluate도 배치 사이즈가 있는데 기본값은  32이다.
 w값이 나와있어 그걸로 연산했기 때문에 의미가 없다. 
'''

y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

import matplotlib.pyplot as plt
plt.plot( hist.history['loss']) # x = epochs, 시간일경우 생략 y hist
plt.plot( hist.history['val_loss'])
plt.title("loss , val_loss")
plt.xlabel('epochs')
plt.ylabel('loss , val_loss')
plt.legend(['train loss', 'val loss']) # 벙례

plt.show()
# 노란색 val_loss 파란색 loss


