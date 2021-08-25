from tensorflow.keras.datasets import fashion_mnist
import numpy as np
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255,horizontal_flip=True,
vertical_flip=False,width_shift_range=0.1,height_shift_range=0.1,rotation_range=5,
zoom_range=0.1,shear_range=0.5,fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

#xy_train = train_datagen.flow_from_directory('../_data/brain/train',target_size=(150,150),batch_size=5
#,class_mode='binary',shuffle=False
#) # 이미지크기고정 셔플 디폴트값 True, 폴더안에있는 데이터 가져옴,파일에서댕겨옴
# x,y가 튜플형태로 묶여있음

# 그냥 데이터를 댕겨오려면 flow사용, xy가 나눠져있음
augument_size = 100
x_data = train_datagen.flow(
          np.tile(x_train[0].reshape(28*28),augument_size).reshape(-1,28,28,1),
          np.zeros(augument_size),
          batch_size=augument_size,
          shuffle =False
).next()
# 100장으로 늘려줌
# 이터레이터 방식으로 반환
# 순차적인 리스트 구조
print(type(x_data))
print(type(x_data[0]))
print(type(x_data[0][0]))
print(x_data[0][0].shape)
print(x_data[0][1].shape)
print(x_data[0].shape)
print(x_data[1].shape)

'''
.next 전  -> 한번만 실행 한개씩 실행
<class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
<class 'tuple'>
<class 'numpy.ndarray'>
(100, 28, 28, 1) x
(100,) y

.next 후 -> 전부실행
<class 'tuple'>
<class 'numpy.ndarray'>
<class 'numpy.ndarray'>
(28, 28, 1)
(28, 28, 1)
(100, 28, 28, 1)
(100,)
print(x_data[0].shape) x
print(x_data[1].shape) y
차원이 밀린다
'''
import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(49):
     plt.subplot(7,7,i+1)
     plt.axis('off')
     plt.imshow(x_data[0][i], cmap='gray')

plt.show()
#np.tile 함수안에숫자만큼 반복
#augument_size 100
'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
'''
