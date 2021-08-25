import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=1.2,shear_range=0.7,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

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

np.save('./_save/_npy/k59_manwoman_train_x.npy',arr=xy_train[0][0])
np.save('./_save/_npy/k59_manwoman_train_y.npy',arr=xy_train[0][1])
np.save('./_save/_npy/k59_manwoman_test_x.npy',arr=xy_test[0][0])
np.save('./_save/_npy/k59_manwoman_test_y.npy',arr=xy_test[0][1])

