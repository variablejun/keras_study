from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D,Conv2D,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16, VGG19,DenseNet121
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

# 75x75;
denseNet121 = DenseNet121(weights='imagenet',include_top=False,input_shape=(32,32,3))
denseNet121.trainable=True
model = Sequential()
model.add(denseNet121)
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
          걸린시간 228.6021659374237
          loss :  0.9877926707267761
          accuracy :  0.12242606282234192
          GAP
          걸린시간 222.29802083969116
          loss :  0.9760838150978088
          accuracy :  0.8119000196456909
     False
          FC
          걸린시간 92.82200336456299
          loss :  0.9731684327125549
          accuracy :  0.11789431422948837
          GAP
          걸린시간 86.88938784599304
          loss :  0.9547325372695923
          accuracy :  0.6636000275611877
cifar 100
     True
          FC
          걸린시간 231.63308095932007
          loss :  3.7158777713775635
          accuracy :  0.02398168481886387
          GAP
          걸린시간 220.99493050575256
          loss :  2.9693470001220703
          accuracy :  0.5049999952316284
     False
          FC
          걸린시간 92.24769854545593
          loss :  2.599966287612915
          accuracy :  0.021439027041196823
          GAP
          걸린시간 87.29995536804199
          loss :  2.2934985160827637
          accuracy :  0.4212999939918518       
'''