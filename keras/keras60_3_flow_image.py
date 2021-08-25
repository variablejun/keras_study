from tensorflow.keras.datasets import fashion_mnist
import numpy as np
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=False,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=0.1,shear_range=0.5,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)
augment_size = 40000

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

x_augmented = x_augmented.reshape(x_augmented.shape[0],28,28,1)
x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_test = x_test.reshape(x_test.shape[0],28,28,1)




x_augmented = train_datagen.flow(x_augmented,np.zeros(augment_size),batch_size=augment_size,shuffle=False).next()[0] # y값은 그대로 쓸거기 때문에 x값만 뽑아준다

x_train  = np.concatenate((x_train,x_augmented))
y_train  = np.concatenate((y_train,y_augmented))

print(x_train.shape,y_train.shape) #(100000, 28, 28, 1) (100000,)
x_augmented = x_augmented.reshape(40000,28,28,1)
x_train = x_train.reshape(100000,28,28,1)
# 실습 xtrain과 x_augmented를 비교하는 subplot출력

import matplotlib.pyplot as plt
plt.figure(figsize=(20,2))
for i in range(49):
     plt.subplot(7,7,i+1)
     plt.axis('off')
     plt.imshow(x_augmented[0][i], cmap='gray')

plt.show()