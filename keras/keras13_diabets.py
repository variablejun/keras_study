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

x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=66)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


#2.모델 구성
models = Sequential()
models.add(Dense(400, input_shape=(10,)))
models.add(Dense(1000))
models.add(Dense(100))
models.add(Dense(50))
models.add(Dense(20))
models.add(Dense(10))
models.add(Dense(5))
models.add(Dense(2))
models.add(Dense(1))


#3.complie/훈련
models.compile(loss = 'mse', optimizer = 'adam')
models.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, shuffle=0.2) 

#4.평가/예측
loss = models.evaluate(x_test, y_test) 
print('loss : ', loss)


y_pred = models.predict(x_test) 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
'''
loss :  3109.60205078125
r2score  0.5008961235003055

r2score  0.5037277375596032
'''