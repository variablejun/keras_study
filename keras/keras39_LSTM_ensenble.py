import numpy as np
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU, Input
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
[5,6,7],[6,7,8],[7,8,9],[8,9,10],[9,10,11],
[10,11,12],[20,30,40],[30,40,50],[40,50,60]])
 # 묶음단위 timesteps
x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
[50,60,70],[60,70,80],[70,80,90],[80,90,100],[90,100,110],
[100,110,120],[2,3,4],[3,4,5],[4,5,6]])

# 각 timesteps에 feature단위로 
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])
x1_predic = np.array([55,65,75]).reshape(1,3,1)
x2_predic = np.array([65,75,85]).reshape(1,3,1)

 # (13, 3) (13, 3) (13,)

#x = x.reshape(13,3,1)
x1 = x1.reshape(x1.shape[0],x1.shape[1],1)
x2 = x2.reshape(x2.shape[0],x2.shape[1],1)

#inputs: A 3D tensor, with shape [batch, timesteps, feature].(4,3,1) 

#model.add(SimpleRNN(16,activation = 'relu',input_shape=(3,1))) # units RNN, DNN fliter CNN


input1 = Input(shape=(3,1))
xx = LSTM(32,activation = 'relu')(input1)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output1 = Dense(2)(xx)

input2 = Input(shape=(3,1))
xx = LSTM(32,activation = 'relu')(input2)
xx = Dense(16)(xx)
xx = Dense(8)(xx)
xx = Dense(4)(xx)
output2 = Dense(2)(xx)

from tensorflow.keras.layers import concatenate, Concatenate  #  소문자 메소드,  대문자 클래스
merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
last_output1 = Dense(1)(merge3)

model = Model(inputs = [input1, input2], outputs = last_output1)
'''

input1 = Input(shape=(3,)) 
dence1 = Dense(10, activation='relu')(input1)
dence2 = Dense(7, activation='relu')(dence1)
dence3 = Dense(5, activation='relu')(dence2)
output1 = Dense(1)(dence3)


# 2-2

input2 = Input(shape=(3,)) 
dence11 = Dense(10, activation='relu')(input2)
dence12 = Dense(7, activation='relu')(dence11)
dence13 = Dense(5, activation='relu')(dence12)
dence14 = Dense(5, activation='relu')(dence13)
output2 = Dense(1)(dence14)
# output이 나왔지만 히든레이어

from tensorflow.keras.layers import concatenate, Concatenate  #  소문자 메소드,  대문자 클래스
merge1 = concatenate([output1, output2]) # concatenate는 양쪽 모델에서 나온 아웃풋을 하나로묶어준다.
merge2 = Dense(10)(merge1)
merge3 = Dense(10)(merge2)
last_output1 = Dense(1)(merge3)

model = Model(inputs = [input1, input2], outputs = last_output1)
'''

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience=50, mode='min', verbose=1)

model.compile(loss = 'mse', optimizer = 'adam')
model.fit([x1,x2], y, epochs=10000, batch_size=1, callbacks=[es]) 
#inputs = np.array([5,6,7]).reshape(1,3,1)

results = model.predict([x1_predic,x2_predic])
print(results)

'''
[[85.35775]]
'''