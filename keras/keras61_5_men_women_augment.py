import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=1.2,shear_range=0.7,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
augment_size = 40000

xy_train = train_datagen.flow_from_directory('../_data/men_women',target_size=(150,150)
,batch_size=3309
,class_mode='categorical',shuffle=False
)

# 이미지크기고정 셔플 디폴트값 True
#D:\_data\men_women
xy_test = test_datagen.flow_from_directory('../_data/men_women',target_size=(150,150)
,batch_size=3309
,class_mode='categorical'
)
x_predic = test_datagen.flow_from_directory('../_data/jbj',target_size=(150,150),batch_size=5
,class_mode='categorical'
) 

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]
x_predic = x_predic[0]

randidx = np.random.randint(x_train.shape[0], size=augment_size)

print(x_train.shape[0])
print(randidx)
print(randidx.shape)

'''
60000
[17779 40024 48101 ... 58251 28004 47306]
(40000,)
'''

x_augmented = x_train[randidx].copy() # 메모리 공유방지
y_augmented = y_train[randidx].copy()

x_augmented = x_augmented.reshape(x_augmented.shape[0],150,150,3)
x_train = x_train.reshape(x_train.shape[0],150,150,3)
x_test = x_test.reshape(x_test.shape[0],150,150,3)

print(x_augmented[0][0].shape) #(10,)

print(x_augmented[0][1].shape)
print(x_augmented[0][1][:10])



x_augmented = train_datagen.flow(x_augmented,np.zeros(augment_size),batch_size=augment_size
,save_to_dir='d:/temp/'
,shuffle=False).next()[0]
x_train  = np.concatenate((x_train,x_augmented))
y_train  = np.concatenate((y_train,y_augmented))


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