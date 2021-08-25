import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=1.2,shear_range=0.7,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory('../_data/brain/train',target_size=(150,150),batch_size=200
,class_mode='binary',shuffle=False
) # 이미지크기고정 셔플 디폴트값 True

xy_test = test_datagen.flow_from_directory('../_data/brain/test',target_size=(150,150),batch_size=200
,class_mode='binary'
) # 배치사이즈를 키워 넘파이로 저장할수잇도록한다. 0방과 1방에 몰아넣는다.
print(xy_train[0][0]) # x
print(xy_train[0][1]) # y [0][2] 는 없음

print(xy_train[0][0].shape,xy_train[0][1].shape)
# 마지막, 32개
#(5, 150, 150, 3) (5,) batchsize

print(type(xy_train)) # <class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0][0]))#<class 'numpy.ndarray'>

np.save('./_save/_npy/k59_3_train_x.npy',arr=xy_train[0][0])
np.save('./_save/_npy/k59_3_train_y.npy',arr=xy_train[0][1])
np.save('./_save/_npy/k59_3_test_x.npy',arr=xy_test[0][0])
np.save('./_save/_npy/k59_3_test_y.npy',arr=xy_test[0][1])



