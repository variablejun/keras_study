#가장 잘나온 전위학습모델로 해당파일을 학습시켜 기존파일과 결과비교


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
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16, VGG19,DenseNet121
denseNet121 = DenseNet121(weights='imagenet',include_top=False,input_shape=(150,150,3))
model = Sequential()
model.add(denseNet121)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='acc', patience=10, mode='max', verbose=3)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=100,callbacks=[es])
# 
loss = model.evaluate(x_test, y_test) 
y_predic = model.predict([x_predic])

acc = hist.history['acc']
loss = hist.history['loss']

print(acc[-1])

print('iu님은 ',(1-y_predic)*100,'%로 여자입니다.')
'''
0.9926597476005554
iu님은  [[3.1934023]] %로 여자입니다.

0.9715926051139832
iu님은  [[0.00021458]] %로 여자입니다.
'''