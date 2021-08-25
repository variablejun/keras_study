from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 

x_train = x_train.reshape(50000, 32* 32* 3)
x_test = x_test.reshape(10000, 32* 32*3)
from sklearn.preprocessing import OneHotEncoder,StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32,3)




# fully c 모델,False적용해서 풀리커넥이 아닌 경우 , avplool적용


vgg16 = VGG16(weights='imagenet',include_top=False,input_shape=(32,32,3))
vgg16.trainable=True
model = Sequential()
model.add(vgg16)
# model.add(GlobalAveragePooling2D())
model.add(Dense(100))
model.add(Dense(10))
# vgg16을 연산하지않아 가중치에 변화없이 기존에 모델링으로 주는것, 가중치 동결 훈련 동결

import time
starttime = time.time()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=512, validation_split=0.3, verbose=2) 

loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
True
걸린시간 831.7490224838257
loss :  8.059049606323242
accuracy :  0.10000000149011612

False
걸린시간 353.0989089012146
loss :  7.850527286529541
accuracy :  0.20649999380111694

스케일러
True

False
걸린시간 317.0341579914093
loss :  2.1536223888397217
accuracy :  0.37229999899864197

GAP 적용

True

False

'''