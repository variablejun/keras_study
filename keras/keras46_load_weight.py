from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential ,load_model
from tensorflow.keras.layers import Dense, Dropout
datasets = load_diabetes()

x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
scaler.fit(x_train) 
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

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
#model.save('./_save/keras46_1_save_model_1.h5')
model.save_weights('./_save/keras46_1_save_weight_1.h5') # 꽝 의미없음
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es]) 
#model.save('./_save/keras46_1_save_model_2.h5')
model.save_weights('./_save/keras46_1_save_weight_2.h5') # 가중치
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime
y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
얼리스타핑 하고 저장하자.
얼리스타핑의 단점은 페이션스 20번 줫을때 최소값이 나와도 20번째 weight값이 저장된다
그 단점을 매 에포마다 값을 저장해서 해결한게 모델 체크포인트
죽겟어요
'''