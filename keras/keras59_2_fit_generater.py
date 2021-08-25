import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#1.data
train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=1.2,shear_range=0.7,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory('../_data/brain/train',target_size=(150,150),batch_size=5
,class_mode='binary',shuffle=False
) # 이미지크기고정 셔플 디폴트값 True

xy_test = test_datagen.flow_from_directory('../_data/brain/test',target_size=(150,150),batch_size=5
,class_mode='binary'
) 
#print(xy_train[0][0]) # x
#print(xy_train[0][1]) # y [0][2] 는 없음

#print(xy_train[0][0].shape,xy_train[0][1].shape)
#print(xy_train[31][1]) # 마지막, 32개
#(5, 150, 150, 3) (5,) batchsize

#print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
#print(type(xy_train[0][0]))#<class 'numpy.ndarray'>


#2 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,Flatten
model = Sequential()
model.add(Conv2D(32,(2,2),input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
hits = model.fit_generator(xy_train,epochs=50,steps_per_epoch=32, validation_data=xy_test,validation_steps=4)
# xy가 붙어있을경우

acc = hits.history('acc')

val_acc = hits.history('val_acc')
loss = hits.history('loss')
val_loss = hits.history('val_loss')
print(acc[-1])
print(val_acc[-1])





