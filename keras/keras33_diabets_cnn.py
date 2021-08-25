from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, GlobalAveragePooling2D

# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 
# (23, 10) (419, 10)
x_train = x_train.reshape(419,10,1,1) # 이미지 4차원 데이터도 순서변경없이 차원수를 낮춰 DNN연산가능
x_test = x_test.reshape(23, 10, 1,1) # 트레인 사이즈의 크기를 바꿀경우 에러가 난다.

#2.모델 구성
model = Sequential()

model.add(Conv2D(16,3 ,padding = 'same' ,input_shape=(10,1,1)))
model.add(Conv2D(8,3,padding = 'same' ,activation = 'relu'))
#model.add(MaxPooling2D())
model.add(Conv2D(4,3,padding = 'same' ,activation = 'relu'))
'''
valid,model.add(MaxPooling2D()) 적용시 오류

ValueError: Negative dimension size caused by subtracting 2 from 1 for 
'{{node conv2d/Conv2D}} = Conv2D[T=DT_FLOAT, data_format="NHWC", dilations=[1, 1, 1, 1], explicit_paddings=[], padding="VALID", strides=[1, 1, 1, 1], use_cudnn_on_gpu=true](Placeholder, conv2d/Conv2D/ReadVariableOp)'
 with input shapes: [?,10,1,1], [2,2,1,128].
'''
model.add(Flatten())
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

print('loss : ', loss)
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)
print("걸린시간",end)

'''
1D
StandardScaler
loss :  4833.31103515625
r2score  0.3519392198190815
걸린시간 6.860862493515015

RobustScaler
loss :  4639.46923828125
r2score  0.37792988832977814
걸린시간 5.693086624145508

MinMaxScaler
loss :  3625.48193359375
r2score  0.5138875049881018
걸린시간 7.770100831985474

2D
Conv2D(32
loss :  3936.20166015625
r2score  0.4722255363438762
걸린시간 8.421845436096191

Conv2D(16
loss :  3744.376708984375
r2score  0.4979458503797921
걸린시간 6.6011621952056885
'''