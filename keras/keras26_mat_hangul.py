from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
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
train_size = 0.7, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
#scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train) #  
x_train = scaler.transform(x_train) #
x_test = scaler.transform(x_test) # 
'''

'''




#2.모델 구성
model = Sequential()
model.add(Dense(50,activation='relu', input_dim = 13))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(16,activation='relu'))
model.add(Dense(1))


#3.complie/훈련
model.compile(loss = 'mse', optimizer = 'adam')
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=20, mode='min', verbose=1)
'''

'''
hist = model.fit(x_train, y_train, epochs=1000, batch_size=10, validation_split=0.3, callbacks=[es]) 
print(hist.history.keys())
print(hist.history['loss'])
print(hist.history['val_loss'])

'''

'''
#4.평가/예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)
'''

'''

y_pred = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print("r2score ", r2)

import matplotlib.pyplot as plt
import matplotlib

plt.plot( hist.history['loss']) # x = epochs, 시간일경우 생략 y hist
plt.plot( hist.history['val_loss'])
plt.title("로스 , 발로스")
plt.xlabel('epochs')
plt.ylabel('loss , val_loss')
plt.legend(['train loss', 'val loss']) # 벙례
plt.show()
# 노란색 val_loss 파란색 loss
# C:\ProgramData\Anaconda3\lib\site-packages\matplotlib\mpl-data\matplotlibrc
# 설정파일 안에 font familly 를 NanumGothic 으로 바꾸고 unicode minus를 False로 바꾸엇습니다.
# C:\Users\bit\.matplotlib 여기서 캐쉬파일을 지우고 재실행햐엿습니다.
print('설정파일 : ',matplotlib.matplotlib_fname())
print(matplotlib.get_cachedir())


