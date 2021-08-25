import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=True,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=1.2,shear_range=0.7,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

x_predic = test_datagen.flow_from_directory('../_data/jbj',target_size=(150,150),batch_size=5
,class_mode='categorical'
) 
print(x_predic[0][0])


np.save('./_save/_npy/k59_manwoman_predic_x.npy',arr=x_predic[0][0])

