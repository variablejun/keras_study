

# 실습1. 
# 맨 우먼 데이터를  모델링구성

#실습2.
#본인사진으로 predict

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

x_train =  np.load('./_save/_npy/k59_manwoman_train_x.npy')
y_train =  np.load('./_save/_npy/k59_manwoman_train_y.npy')

x_test =  np.load('./_save/_npy/k59_manwoman_test_x.npy')
y_test =  np.load('./_save/_npy/k59_manwoman_test_y.npy')
x_predic =  np.load('./_save/_npy/k59_manwoman_predic_x.npy')



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten
model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(2,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=100,steps_per_epoch=32, validation_split=0.3,validation_steps=4)
# 
loss = model.evaluate(x_test, y_test) 
y_predic = model.predict([x_predic])
acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
print(acc[-1])
print(val_acc[-1])
print('iu님은 ',(1-y_predic)*100,'%로 여자입니다.')
'''
0.9982728958129883 acc
0.4602215588092804
iu님은  [[ 0.29190183 99.7298    ]] %로 여자입니다.
'''