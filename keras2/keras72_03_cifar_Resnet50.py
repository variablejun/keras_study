from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19,ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
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


resNet50 = ResNet50(weights='imagenet',include_top=False,input_shape=(32,32,3))
resNet50.trainable=True
model = Sequential()
model.add(resNet50)
#model.add(GlobalAveragePooling2D())
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
          걸린시간 194.51387906074524
          loss :  1.3099935054779053
          accuracy :  0.12109062075614929
          GAP
          걸린시간 188.49628901481628
          loss :  1.2473509311676025
          accuracy :  0.7617999911308289
     False
          FC
          걸린시간 73.88782048225403
          loss :  1.572860598564148
          accuracy :  0.11098253726959229
          GAP
          걸린시간 69.6081326007843
          loss :  1.5658159255981445
          accuracy :  0.4514000117778778         
cifar 100

     True
          FC
          걸린시간 193.9466438293457
          loss :  4.173280239105225
          accuracy :  0.022852657362818718
          GAP
          걸린시간 188.4764986038208
          loss :  2.9538300037384033
          accuracy :  0.4814000129699707     
     False
          FC
          걸린시간 73.80445194244385
          loss :  3.7551867961883545
          accuracy :  0.015271591953933239
          GAP
          걸린시간 69.8150646686554
          loss :  3.5901544094085693
          accuracy :  0.18160000443458557
'''