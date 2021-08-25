import numpy as np
# data
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,11,8,9,10])

# model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(1000, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

# compile

from tensorflow.keras.optimizers import Adam,Adadelta,Adamax,Adagrad
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

optimizer = SGD(lr=0.01,nesterov=True)
'''
Adam default 0.001
lr=0.1
loss : 1.8303501605987549 pred : [[11.283883]]
lr=0.01
loss : 253.2394256591797 pred : [[0.8295814]]
lr=0.001
loss : 1.4594148397445679 pred : [[12.13138]]

Adadelta default 1.0
lr=0.1
loss : 1.5136511325836182 pred : [[11.160503]]
lr=0.01
loss : 1.4088900089263916 pred : [[11.558767]]
lr=0.001
loss : 10.038787841796875 pred : [[6.532822]]

Adamax default 0.002
lr=0.1
loss : 1.4131563901901245 pred : [[11.523305]]
lr=0.01
loss : 1.541235327720642 pred : [[11.14854]]
lr=0.001
loss : 1.880216360092163 pred : [[10.527557]]

Adagrad default 0.01
lr=0.1
loss : 2.6892497539520264 pred : [[13.665975]]
lr=0.01
loss : 1.4364681243896484 pred : [[11.424371]]
lr=0.001
loss : 1.4140737056732178 pred : [[11.525011]]

RMSprop default 0.001
lr=0.1
loss : 447811776.0 pred : [[30615.857]]
lr=0.01
loss : 12.966775894165039 pred : [[4.663063]]
lr=0.001
loss : 1.6552555561065674 pred : [[11.034265]]

SGD default 0.01
lr=0.1
loss : nan pred : [[nan]]
lr=0.01
loss : nan pred : [[nan]]
lr=0.001
loss : 1.9773740768432617 pred : [[10.311184]]

Nadam default0.002
lr=0.1
loss : 320.121826171875 pred : [[-17.884262]]
lr=0.01
loss : 1.8969606161117554 pred : [[10.360258]]
lr=0.001
loss : 1.4010812044143677 pred : [[11.653926]]

'''
model.compile(loss='mse', optimizer=optimizer,metrics=['mse'])
model.fit(x,y,epochs=100,batch_size=1)

loss, mse = model.evaluate(x,y)
y_predict = model.predict([11])
print('loss :', loss, 'pred :', y_predict)

'''
기본
loss : 4.50625921075698e-05 pred : [[11.013466]]
값 바꿈
loss : 6.525688171386719 pred : [[11.608394]]
0.1
loss : 7.197054386138916 pred : [[10.765977]]
0.01
loss : 6.526180267333984 pred : [[11.773119]]
0.001
loss : 6.525956630706787 pred : [[11.961805]]
'''



