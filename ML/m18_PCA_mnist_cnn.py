# mnist pca cnn 으로 구성 28 *28 -> 784

import numpy as np
from sklearn.decomposition import PCA
from tensorflow.keras.layers import Dense,Flatten,Conv2D,MaxPooling2D
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data() #_ 무시하겟다 받지않겟다.
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test,axis=0)
y = np.append(y_train, y_test,axis=0)
print(x.shape) # (70000, 28, 28)
 # (70000, 28, 28)

pca = PCA(n_components=400) # 랜덤이 아닌 자기 기준에 따라서 압축을 시킨다 -> 제거하는것은 아님
x = x.reshape(70000,28*28)
#(70000, 400)
x = pca.fit_transform(x)
x = x.reshape(70000,20,20,1)
print(x.shape)
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

# 0.95 이상의 컴포넌트
# 텐서 Dnn으로 구성 기존 DNN과비교
# 기존  784개에서 몇개줄어드는지

pcaEVR = pca.explained_variance_ratio_
 
cunsum = np.cumsum(pcaEVR)

from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() # 리스트를 배열로 바꾸어주는 함수
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()



model = Sequential() 
model.add(Conv2D(50, kernel_size=(2,2) ,padding = 'same' ,input_shape=(20, 20, 1)))
model.add(Conv2D(15, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(3, (2,2),padding = 'same' , activation = 'relu'))
model.add(Conv2D(2, (2,2) ,activation = 'relu'))

model.add(MaxPooling2D())
model.add(Conv2D(2, (2,2),padding = 'same' ,activation = 'relu'))
model.add(Conv2D(2, (2,2),padding = 'same' ,activation = 'relu'))
model.add(MaxPooling2D())
model.add(Flatten())#(N, 180)

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='sigmoid')) # 원핫 인코딩을하면 배열로 특성있는 부분이 펴지면서 바뀐다

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience=50, mode='max', verbose=3)

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=1000, batch_size=256, validation_split=0.3, verbose=1,callbacks=[es]) 
loss = model.evaluate(x_test, y_test) 

print('loss : ', loss[0])
print('accuracy : ', loss[1])
'''
dnn

loss :  0.20692114531993866
accuracy :  0.977400004863739

cnn
loss :  0.1713060885667801
accuracy :  0.9821000099182129

pca
loss :  0.38625866174697876
accuracy :  0.9712380766868591
'''