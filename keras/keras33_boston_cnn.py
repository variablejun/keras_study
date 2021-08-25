
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x = datasets.data
y = datasets.target
'''

'''
print(np.max(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) # x train를 스케일 하여 스케일링 기준을 xtrain으로 만들고 
x_train = scaler.transform(x_train) #
x_test = scaler.transform(x_test) # xtrain 기준의 scaler를 test에 적용시킨다

x_train = x_train.reshape(480,13,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(26, 13, 1,1)

'''

'''
# 열마다 MinMaxScaler로 전처리를 해서 모든 데이터 전처리했을때보다 성능을더 좋게 만든다
#print(x_scale[:])
#print(np.max(x_scale), np.min(x_scale)



#2.모델 구성 1D이용시 차원수부족 오류
model = Sequential() 
model.add(Conv2D(26,3 ,padding = 'same' ,input_shape=(13,1,1)))
model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())#(N, 180)
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=64, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime

print("걸린시간", end)
print('loss : ', loss)
y_pred = model.predict(x_test) 

# y_pred = scaler.transform(y_pred)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
걸린시간 10.174026489257812
loss :  55.307334899902344
r2score  0.5710622971648347

모델링 수정후

걸린시간 17.49739170074463
loss :  13.537178993225098
r2score  0.8950120009397117

걸린시간 19.379173040390015
loss :  15.343551635742188
r2score  0.8810026321765818

validation_split=0.003
걸린시간 11.505271196365356
loss :  6.488640308380127
r2score  0.9496771549474132

2D
걸린시간 6.761014938354492
loss :  23.282869338989258
r2score  0.8194289855347043

model = Sequential() 
model.add(Conv2D(26,3 ,padding = 'same' ,input_shape=(13,1,1)))

model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())#(N, 180)

걸린시간 4.68802547454834
loss :  13.895647048950195
r2score  0.892231884416774


model = Sequential() 
model.add(Conv2D(26,3 ,padding = 'same' ,input_shape=(13,1,1)))
model.add(Conv2D(16,3,padding = 'same' ,activation = 'relu'))
model.add(Conv2D(1,3,padding = 'same' ,activation = 'relu'))
model.add(Flatten())#(N, 180)
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))

걸린시간 12.57477617263794
loss :  8.286820411682129
r2score  0.9357313132695979
'''