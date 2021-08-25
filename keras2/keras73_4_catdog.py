#이진분류지만 다중분류사용 ->softmax와 categorycal

#categoricalb + sigmoid


import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator



x_train =  np.load('./_save/_npy/k59_catdog_train_x.npy')
y_train =  np.load('./_save/_npy/k59_catdog_train_y.npy')
x_test =  np.load('./_save/_npy/k59_catdog_test_x.npy')
y_test =  np.load('./_save/_npy/k59_catdog_test_y.npy')



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten,GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import VGG16, VGG19,DenseNet121
denseNet121 = DenseNet121(weights='imagenet',include_top=False,input_shape=(150,150,3))
denseNet121.trainable=True
model = Sequential()
model.add(denseNet121)
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))

es = EarlyStopping(monitor='acc', patience=10, mode='max', verbose=3)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hist = model.fit(x_train,y_train,epochs=100,batch_size=32,callbacks=[es])
# 
loss = model.evaluate(x_test, y_test) 

acc = hist.history['acc']
loss = hist.history['loss']

print(acc[-1])


'''
전
0.7139032483100891
후
0.992129921913147
'''