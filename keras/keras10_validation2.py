from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

x_train = x[:7]
y_train = y[:7]
x_test = x[7:10]

y_test = y[7:10]
x_val = x[10:13]
y_val = y[10:13]
'''
print(np.shape(x_train), x_train)
print(np.shape(y_train), y_train)
print(np.shape(x_test), x_test)
print(np.shape(y_test), y_test)
print(np.shape(x_val), x_val)
print(np.shape(y_val), y_val)
'''
# 나누기

#x_train = np.array([1,2,3,4,5,6,7]) # 공부
#y_train = np.array([1,2,3,4,5,6,7]) 
#x_test = np.array([8,9,10])  #평가
#y_test = np.array([8,9,10]) 
#x_val = np.array([11,12,13])
#y_val = np.array([11,12,13])

model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1)) 
model.add(Dense(3)) 
model.add(Dense(4))


model.compile(loss = 'mse', optimizer = 'adam')
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val,y_val)) 

loss = model.evaluate(x_test,y_test) # 데이터를 훈련데이터와 평가데이터를 분리하여 다른걸 사용한다. -> 과적합을 방지하기 위해서
print("LOSS : ",loss)
y_predict = model.predict([11]) #


"""
7/7 [==============================] - 0s 4ms/step - loss: 1.8868 - val_loss: 14.3894
예측값의 대한 로스
val_loss 컴퓨터 내부에서 판단한 로스, 기존의 로스는 과적합되는 경우가 많아 val_loss 를 중점에 둬야한다

plt.scatter(x,y) # 원 좌표
plt.plot(x, y_predict, color='red') # 예측한 좌표
plt.show()


"""

