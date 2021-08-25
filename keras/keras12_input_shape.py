
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3],[10,9,8,7,6,5,4,3,2,1]]) # 3, 10
x = np.transpose(x) # 10 ,3 
y = np.array([11,12,13,14,15,16,17,18,19,20]) 
# (10,)

model = Sequential()
# model.add(Dense(10, input_dim=3)) #-> x의 열
model.add(Dense(10, input_shape=(3,))) # 3차원이상일경우 사용
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(1))

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=10,batch_size=1)

loss = model.evaluate(x,y)
print('loss : ', loss)

x_pred = np.array([[10, 1.3, 1]])
result = model.predict(x_pred)

print('10, 1.3, 1의 예측값 : ', result)

'''

'''
