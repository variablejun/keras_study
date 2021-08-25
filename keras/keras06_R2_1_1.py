
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt
import random


x = np.array(range(100)) # 0 ~ 99
y = np.array(range(1,101))  # 1 ~ 100

x_train = x[:70]
y_train = y[:70]
x_test = x[-30:]
y_test = y[70:]

model = Sequential()
model.add(Dense(1, input_dim = 1))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
test_size = 0.7, random_state=66) 

loss = model.evaluate(x_test, y_test) # x의 대한 테스트 
print('loss : ', loss)

y_pred = np.array([100])

result = model.predict(x_test)
print('100의 예측값 : ', result)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, result) # y의원래값과 y의 예측값
print('100의 예측값 : ', r2)
"""
결정계수 (R2) : 회기모델의 지표, max= 1 , min = 0 
sklearn shuffle은 기본적으로 True


"""

