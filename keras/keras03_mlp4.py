
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax



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

model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=100,batch_size=1)

result = model.predict(x)

#ax.plot(x,y,result,color = 'blue')

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot3(result,y,color='red')

plt.scatter(x, y[:,0])
plt.scatter(x, y[:,1])
plt.scatter(x, y[:,2])


plt.show()


'''

ax.scatter3D(x,y[:,0])
ax.scatter3D(x,y[:,1])
ax.scatter3D(x,y[:,2])


loss = model.evaluate(x,y)
print('loss : ', loss)

x_pred = np.array([[9]]) #  9에 대한 예측값(y)

result = model.predict(x_pred)

print('10, 1.3, 1의 예측값 : ', result)


10000
loss :  0.005376973655074835
10, 1.3, 1의 예측값 :  [[10.016304   1.5323956  1.0061471]]

1000
loss :  0.009408180601894855
10, 1.3, 1의 예측값 :  [[9.851324   1.5280446  0.95864725]]

900
loss :  0.005425484385341406
10, 1.3, 1의 예측값 :  [[10.017113   1.528147   1.0045078]]

'''
