from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, Dropout

# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']



from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 


#2.모델 구성
model = Sequential()
model.add(Dense(256,  activation='relu', input_shape=(10,)))

model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1))

#model = load_model('./_save/keras46_1_save_model_1.h5') # 작동 모델만 세이브
# 모델만 저장되어있어 반복할때마다 값이 다르다.

#model = load_model('./_save/keras46_1_save_model_2.h5') # 모델과 fit compile세이브
# 이미 fit에서 나온 가중치가 저장되어있어 반복적으로 실행하여도 똑같은 값이 나온다.

#4.평가/예측
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True,mode='auto', filepath='./_save/ModelCheckpoint/keras47_MCP.hdf5')

import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
# model.load_weights('./_save/keras46_1_save_weight_2.h5') # save_weights에는 위치에 상관없이 모델이 저장이 되지 않는다.
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es, cp]) 
model.save('./_save/ModelCheckpoint/keras47_Model_Save.h5')

loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime



y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
상황에맞춰서쓴다
1/1 [==============================] - 0s 448ms/step - loss: 4595.1064
r2score  0.3838782183383098

1/1 [==============================] - 0s 451ms/step - loss: 4595.1064
r2score  0.3838782183383098


Epoch 00033: early stopping
1/1 [==============================] - 0s 12ms/step - loss: 4159.4995
r2score  0.44228530768161367
'''