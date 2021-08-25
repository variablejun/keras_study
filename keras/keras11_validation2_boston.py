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
test_size = 0.7, shuffle=True ,random_state=66)

#2.모델 구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))


#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train, y_train, epochs=30, batch_size=1, validation_split=0.2, shuffle=0.2) 

#4.평가/예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
result = model.predict([10]) 
print('price: ', result)