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
train_size = 0.7, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


#2.모델 구성
model = Sequential()
model.add(Dense(400,activation='relu', input_shape=(10,)))
model.add(Dense(50,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(5,activation='relu'))
model.add(Dense(2,activation='relu'))
model.add(Dense(1))


#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=100, batch_size=10, validation_split=0.3) 

#4.평가/예측
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss)
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
StandardScaler
loss :  3017.287841796875
r2score  0.5157129350932576

StandardScaler 활성함수 추가후 결과
loss :  4179.9189453125
r2score  0.32910587578655437

MinMaxScaler
loss :  2986.849609375
r2score  0.5205983543579304

MinMaxScaler 활성함수 추가후 결과
loss :  3139.98046875
r2score  0.4960202531973571

기존 스코어
loss :  3109.60205078125
r2score  0.5008961235003055

'''