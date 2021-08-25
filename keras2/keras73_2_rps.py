2#categoricalb + sigmoid


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, VGG19,DenseNet121
x_train =  np.load('./_save/_npy/rps_train_x.npy')
y_train =  np.load('./_save/_npy/rps_train_y.npy')
x_test =  np.load('./_save/_npy/rps_test_x.npy')
y_test =  np.load('./_save/_npy/rps_test_y.npy')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
model = Sequential()

denseNet121 = DenseNet121(weights='imagenet',include_top=False,input_shape=(150,150,3))
denseNet121.trainable=True
model = Sequential()
model.add(denseNet121)
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))

es = EarlyStopping(monitor='acc', patience=10, mode='max', verbose=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=100,callbacks=[es])
# 
loss = model.evaluate(x_test, y_test) 

acc = hist.history['acc']

loss = hist.history['loss']

print(acc[-1])

'''
전이학습 전 acc: 0.9926303625106812

전이학습 후 acc : 0.9408730268478394
'''