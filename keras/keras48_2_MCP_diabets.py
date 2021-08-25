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

#model = load_model('./_save/keras46_1_save_model_1.h5') # 작동 모델만 세이브
# 모델만 저장되어있어 반복할때마다 값이 다르다.

#model = load_model('./_save/keras46_1_save_model_2.h5') # 모델과 fit compile세이브
# 이미 fit에서 나온 가중치가 저장되어있어 반복적으로 실행하여도 똑같은 값이 나온다.
'''
#4.평가/예측
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True,mode='auto', filepath='./_save/ModelCheckpoint/keras47_MCP.hdf5')
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
# model.load_weights('./_save/keras46_1_save_weight_2.h5') # save_weights에는 위치에 상관없이 모델이 저장이 되지 않는다.
model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es]) 
model.save('./_save/ModelCheckpoint/keras47_Model_Save.h5')
'''
#model = load_model('./_save/ModelCheckpoint/keras47_Model_Save.h5')
model = load_model('./_save/ModelCheckpoint/keras47_MCP.hdf5')
loss = model.evaluate(x_test, y_test,batch_size=64) 

y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''

로드 모델(model + 가중치)
1/1 [==============================] - 0s 461ms/step - loss: 4159.4995
r2score  0.44228530768161367

체크포인트
1/1 [==============================] - 0s 453ms/step - loss: 4512.4692
r2score  0.3949583657774144

'''