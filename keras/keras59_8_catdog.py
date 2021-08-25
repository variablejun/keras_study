#categoricalb + sigmoid


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train =  np.load('./_save/_npy/k59_catdog_train_x.npy')
y_train =  np.load('./_save/_npy/k59_catdog_train_y.npy')
x_test =  np.load('./_save/_npy/k59_catdog_test_x.npy')
y_test =  np.load('./_save/_npy/k59_catdog_test_y.npy')




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=100,steps_per_epoch=32, validation_split=0.3,validation_steps=4)
# 
loss = model.evaluate(x_test, y_test) 

acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
print(acc[-1])
print(val_acc[-1])

'''
0.2860967218875885
1.0

0.7139032483100891
0.0
'''