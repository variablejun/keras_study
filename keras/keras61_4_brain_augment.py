'''
np.save('./_save/_npy/k59_3_train_x.npy',arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_train_y.npy',arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_test_x.npy',arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_test_y.npy',arr=xy_test[0][1])


'''
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train =  np.load('./_save/_npy/k59_3_train_x.npy')
y_train =  np.load('./_save/_npy/k59_3_train_y.npy')
x_test =  np.load('./_save/_npy/k59_3_test_x.npy')
y_test =  np.load('./_save/_npy/k59_3_test_y.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hits = model.fit(x_train,y_train,epochs=50,steps_per_epoch=32, validation_split=0.3,validation_steps=4)
# 
loss = model.evaluate(x_test, y_test) 

acc = hits.history['acc']
val_acc = hits.history['val_acc']
loss = hits.history['loss']
val_loss = hits.history['val_loss']
print(acc[-1])
print(val_acc[-1])