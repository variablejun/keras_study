from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19,DenseNet121,MobileNetV2
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
mobileNetV2 = MobileNetV2(weights='imagenet',include_top=False,input_shape=(32,32,3))
mobileNetV2.trainable=True
model = Sequential()
model.add(mobileNetV2)
model.add(GlobalAveragePooling2D())

# model.add(Dense(1024))
# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dense(128))

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
          걸린시간 96.49979281425476
          loss :  6.923775672912598
          accuracy :  0.10685110837221146
          GAP
          걸린시간 89.11820840835571
          loss :  6.152239799499512
          accuracy :  0.46160000562667847
     False
          FC
          걸린시간 55.63110065460205
          loss :  2.0961976051330566
          accuracy :  0.10376738756895065
          GAP
          걸린시간 49.72766351699829
          loss :  2.0768227577209473
          accuracy :  0.24379999935626984
cifar 100
     True
          FC
          걸린시간 93.32697892189026
          loss :  20.145124435424805
          accuracy :  0.012500625103712082
          GAP
          걸린시간 87.54523181915283
          loss :  16.68414306640625
          accuracy :  0.042899999767541885
     False
          FC
          걸린시간 55.612056255340576
          loss :  4.4981584548950195
          accuracy :  0.011725004762411118
          GAP
          걸린시간 49.104865312576294
          loss :  4.315945148468018
          accuracy :  0.06989999860525131    
'''