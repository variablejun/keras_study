from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt




x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
y = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
test_size = 0.8, shuffle=True ,random_state=66) # 12개train 3개 test



model = Sequential()
model.add(Dense(5, input_dim=1)) 
model.add(Dense(1)) 
model.add(Dense(3)) 
model.add(Dense(4))


model.compile(loss = 'mse', optimizer = 'adam')
# model.fit(x_train,y_train, epochs=100, batch_size=1, validation_data=(x_val,y_val)) 
model.fit(x_train,y_train, epochs=100, batch_size=1, validation_split=0.3 ,shuffle=True) #  train 의 12개 중에서 0.3개 를 validation 으로 주겟다.

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

