
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


x = np.array([range(10), range(21,31), range(201,211)]) 
x = np.transpose(x)

y = np.array([
			[1,2,3,4,5,6,7,8,9,10], 
			[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], 
			[10,9,8,7,6,5,4,3,2,1]
			])
y = np.transpose(y)

model = Sequential()
model.add(Dense(10, input_dim=3))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(3)) # output dim 3개가 나오기 때문에 바꿔주어야하나

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=10000,batch_size=1)

loss = model.evaluate(x,y)
print('loss : ', loss)

x_pred = np.array([[0, 21, 201]])

result = model.predict(x_pred)

print('1, 1, 10의 예측값 : ', result)

'''
1500
loss :  0.02867881953716278
1, 1, 10의 예측값 :  [[0.9857105 1.5215583 9.980042 ]]

1000
loss :  0.008304745890200138
1, 1, 10의 예측값 :  [[0.9844672 1.1177711 9.949358 ]]

900
loss :  0.03752119839191437
1, 1, 10의 예측값 :  [[0.9649975 1.0662653 9.710153 ]]


10000
loss :  0.00573700712993741
1, 1, 10의 예측값 :  [[ 1.0122045  1.1343617 10.023673 ]]
'''
