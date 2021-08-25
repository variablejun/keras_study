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

model = load_model('./_save/keras46_1_save_model_2.h5') # 모델과 fit compile세이브
# 이미 fit에서 나온 가중치가 저장되어있어 반복적으로 실행하여도 똑같은 값이 나온다.
'''
#4.평가/예측
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=3)
import time
starttime = time.time()
model.compile(loss = 'mse', optimizer = 'adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=16, validation_split=0.003, verbose=2,callbacks=[es]) 
loss = model.evaluate(x_test, y_test,batch_size=64) 
end = time.time()- starttime
'''


y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

'''
loss :  4082.570068359375
r2score  0.4526001610145962
걸린시간 4.525232791900635

load model 후

loss :  5154.60693359375
r2score  0.3088591512340907
걸린시간 4.407160520553589


loss :  4351.6298828125
r2score  0.4165240769583266
걸린시간 4.702455759048462

r2score  0.17935768572950406


r2score  0.17935768572950406

'''