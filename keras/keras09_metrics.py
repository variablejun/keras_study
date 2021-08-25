
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
import time


x = np.array([range(10)]) 
x = np.transpose(x)

y = np.array([
			[1,2,3,4,5,6,7,8,9,10], 
			[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3], 
			[10,9,8,7,6,5,4,3,2,1]
			])

y = np.transpose(y)
ascending = 0

descending = 0

model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(2))
model.add(Dense(3)) # output dim 3개가 나오기 때문에 바꿔주어야하나
start = time.time()
model.compile(loss="mse",optimizer="adam", metrics="mse") #
model.fit(x,y,epochs=1000,batch_size=1,verbose=3)

end = time.time() - start
print('걸린 시간 : ', end)

loss = model.evaluate(x,y)
print('Loss : ', loss)

x_pred = model.predict(x)
print('예측값 : ', x_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y, x_pred) # y의원래값과 y의 예측값

print('r2 score : ', r2)

'''
mea
실제값과 예측값의 차이를 절대값으로 내어주는것
rmse
루트씌움
'''
'''

vervose 수차에 따른 시간변화
0
걸린 시간 :  17.329035997390747

1
10/10 [==============================] - 0s 2ms/step - loss: 0.0082
걸린 시간 :  25.476412296295166

2
10/10 - 0s - loss: 0.0063
걸린 시간 :  20.17378854751587

3
Epoch 1000/1000
걸린 시간 :  20.094128131866455

'''
'''
loss = model.evaluate(x,y)
print('Loss : ', loss)

x_pred = model.predict(x)
print('예측값 : ', x_pred)

from sklearn.metrics import r2_score
r2 = r2_score(y, x_pred) # y의원래값과 y의 예측값

print('r2 score : ', r2)


fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot3(result,y,color='red')

plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])


plt.show()

'''