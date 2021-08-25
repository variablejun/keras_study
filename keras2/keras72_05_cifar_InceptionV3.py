from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19,ResNet101,InceptionV3
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
#(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
print(x_train.shape)
print(x_test.shape)
x_train = x_train.reshape(50000, 32* 32* 3)
x_test = x_test.reshape(10000, 32* 32*3)
from sklearn.preprocessing import OneHotEncoder,StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32,3)




# fully c 모델,False적용해서 풀리커넥이 아닌 경우 , avplool적용

# 75x75;
inceptionV3 = InceptionV3(weights='imagenet',include_top=False,input_shape=(32,32,3))
inceptionV3.trainable=True
model = Sequential()
model.add(inceptionV3)
#model.add(GlobalAveragePooling2D())
model.add(Dense(1024))
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(128))

model.add(Dense(100, activation='softmax'))
# vgg16을 연산하지않아 가중치에 변화없이 기존에 모델링으로 주는것, 가중치 동결 훈련 동결

import time
starttime = time.time()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.3, verbose=2) 

loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
cifar 10
     True
          FC
          걸린시간 320.7445683479309
          loss :  1.5559114217758179
          accuracy :  0.11739391833543777
          GAP
          걸린시간 311.6819076538086
          loss :  1.320055603981018
          accuracy :  0.7580999732017517
     False
          FC
          걸린시간 73.88782048225403
          loss :  1.572860598564148
          accuracy :  0.11098253726959229
          GAP
          걸린시간 115.95006251335144
          loss :  1.7834739685058594
          accuracy :  0.36570000648498535        
cifar 100

     True
          FC
          걸린시간 193.9466438293457
          loss :  4.173280239105225
          accuracy :  0.022852657362818718
          GAP
          걸린시간 309.6808829307556
          loss :  3.5123512744903564
          accuracy :  0.4269999861717224   
     False
          FC
          걸린시간 73.80445194244385
          loss :  3.7551867961883545
          accuracy :  0.015271591953933239
          GAP
          걸린시간 115.5123450756073
          loss :  3.9713873863220215
          accuracy :  0.11680000275373459
'''