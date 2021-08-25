from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt



x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 


x_train = x[:7]
y_train = y[:7]
x_test = x[7:]
y_test = y[7:]
print(x_train)
print(y_train)
print(x_test)
print(y_test)

"""

x_train = np.split(x,1)
y_train = np.split(y,1)
x_test = np.split(x,[7,10])[1]
y_test= np.split(y,[7,10])[1]

model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1)) 
model.add(Dense(3)) 
model.add(Dense(4))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=10000, batch_size=1) 
loss = model.evaluate(x_test,y_test) # 데이터를 훈련데이터와 평가데이터를 분리하여 다른걸 사용한다. -> 과적합을 방지하기 위해서
print("LOSS : ",loss)
y_predict = model.predict([11]) #




"""

