from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19,EfficientNetB0
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

# ValueError: When setting `include_top=True` and loading `imagenet` weights, `input_shape` should be (224, 224, 3).
efficientNetB0 = EfficientNetB0(weights='imagenet',include_top=False,input_shape=(32,32,3))
efficientNetB0.trainable=True
model = Sequential()
model.add(efficientNetB0)
model.add(GlobalAveragePooling2D())

# model.add(Dense(1024))
# model.add(Dense(512))
# model.add(Dense(256))
# model.add(Dense(128))

model.add(Dense(10, activation='softmax'))
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
          걸린시간 164.20868158340454
          loss :  7.025494575500488
          accuracy :  0.2232999950647354
          GAP
          걸린시간 161.77889847755432
          loss :  6.5745415687561035
          accuracy :  0.14380000531673431
     False
          FC
          걸린시간 78.7468478679657
          loss :  2.1946940422058105
          accuracy :  0.16339999437332153
          GAP
          걸린시간 76.83880138397217
          loss :  2.233083486557007
          accuracy :  0.16380000114440918
cifar 100
     True
          FC
          걸린시간 164.346177816391
          loss :  8.417255401611328
          accuracy :  0.18729999661445618
          GAP
          걸린시간 162.07357358932495
          loss :  8.755816459655762
          accuracy :  0.05920000001788139
     False
          FC
          걸린시간 78.9923152923584
          loss :  4.49911642074585
          accuracy :  0.023099999874830246
          GAP
          걸린시간 77.14397263526917
          loss :  4.569753170013428
          accuracy :  0.01510000042617321   
'''