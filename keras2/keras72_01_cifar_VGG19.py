from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
#(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
(x_train, y_train), (x_test, y_test) = cifar100.load_data() 
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


vgg19 = VGG19(weights='imagenet',include_top=False,input_shape=(32,32,3))
vgg19.trainable=True
model = Sequential()
model.add(vgg19)
#model.add(GlobalAveragePooling2D())
model.add(Dense(512))
model.add(Dense(128))

model.add(Dense(100, activation='softmax'))
# vgg16을 연산하지않아 가중치에 변화없이 기존에 모델링으로 주는것, 가중치 동결 훈련 동결

import time
starttime = time.time()
model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10, batch_size=512, validation_split=0.3, verbose=2) 

loss = model.evaluate(x_test, y_test) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
cifar 10
     True
          FC
          걸린시간 423.8399202823639
          loss :  5.30305814743042
          accuracy :  0.09992994368076324
          GAP
          걸린시간 421.274311542511
          loss :  9.66848087310791
          accuracy :  0.10000000149011612
     False
          FC
          걸린시간 438.6036927700043
          loss :  3.6742327213287354
          accuracy :  0.10003002732992172
          GAP
          걸린시간 166.83230209350586
          loss :  2.0166478157043457
          accuracy :  0.44130000472068787          
cifar 100

     True
          FC
          걸린시간 95.10039401054382
          loss :  3.2524428367614746
          accuracy :  0.01482435967773199
          GAP
          걸린시간 93.84857296943665
          loss :  2.937375068664551
          accuracy :  0.24250000715255737          
     False
          FC
          걸린시간 41.46545147895813
          loss :  2.4740171432495117
          accuracy :  0.020922988653182983
          GAP
          걸린시간 40.63818144798279
          loss :  2.533377170562744
          accuracy :  0.3628999888896942
'''