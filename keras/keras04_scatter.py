
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt

x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,4,3,5,7,9,3,8,12]) 


model = Sequential()
model.add(Dense(5, input_dim=1)) 

model.add(Dense(1)) 
model.add(Dense(3)) 
model.add(Dense(4))
model.add(Dense(5))
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')

model.fit(x, y, epochs=3000, batch_size=1) 
y_predict = model.predict(x) # 

plt.scatter(x,y) # 원 좌표
plt.plot(x, y_predict, color='red') # 예측한 좌표
plt.show()

"""



"""

