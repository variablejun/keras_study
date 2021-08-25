from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from keras import optimizers

import time
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=76,shuffle=True)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
start = time.time()

#2.모델 구성
models = Sequential()
models.add(Dense(400, activation='relu',input_shape=(10,)))
models.add(Dense(30))
models.add(Dense(20))
models.add(Dense(60, activation='relu'))
models.add(Dense(20))
models.add(Dense(1))

'''
models.add(Dense(40,activation='relu',input_shape=(10,)))
models.fit(x_train, y_train, epochs=70, batch_size=32, validation_split=0.07, verbose=2) 
r2score  0.5221540889935256
걸린 시간 :  4.4299476146698

models.fit(x_train, y_train, epochs=3, batch_size=10, validation_split=0.01, verbose=2) 
models.add(Dense(2000,activation='relu',input_shape=(10,)))
models.add(Dense(4000, activation='relu'))
r2score  0.5201247879006943
걸린 시간 :  2.470201253890991

models.fit(x_train, y_train, epochs=5, batch_size=10, validation_split=0.01, verbose=2) 
models.add(Dense(1500,activation='relu',input_shape=(10,)))
models.add(Dense(2000, activation='relu'))
models.add(Dense(100, activation='relu'))
r2score  0.5213302094962526
걸린 시간 :  2.600926399230957

epochs=5
models.add(Dense(5000,activation='relu',input_shape=(10,)))
models.add(Dense(10000, activation='relu'))
r2score  0.5224069667180742
걸린 시간 :  3.6801133155822754

models.add(Dense(500,activation='relu',input_shape=(10,)))
models.add(Dense(1000, activation='relu'))
r2score  0.5225580208449581
걸린 시간 :  3.0673816204071045

epochs=10
models.add(Dense(500,activation='relu',input_shape=(10,)))
r2score  0.5175525507914567
걸린 시간 :  3.112518548965454
apochs 횟수와 노드의 연산수는 반비례

models.add(Dense(50,activation='relu',input_shape=(10,)))
r2score  0.5194252035940217
걸린 시간 :  11.81020998954773

validation_split=0.03,
 metrics="mse"
r2score  0.5200815080678738
걸린 시간 :  11.862171411514282

'''
#3.complie/훈련

# sgd = optimizers.SGD(lr = 0.01)

models.compile(loss = 'mse', optimizer ='adam',  metrics="mse")
models.fit(x_train, y_train, epochs=10000, batch_size=8, validation_split=0.2, verbose=2, shuffle=True) 

#4.평가/예측
loss = models.evaluate(x_test, y_test) 
print('loss : ', loss)


y_pred = models.predict(x_test) 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
end = time.time() - start
print('걸린 시간 : ', end)
'''
0.62 이상올리기 과제
r2score  0.5037277375596032
'''