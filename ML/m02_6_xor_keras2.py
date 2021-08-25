from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
#1 data
x_data = [[0,0],[0,1],[1,0],[1,1]] # 2,4

y_data = [0,1,1,0]

#2 model

model = Sequential()
model.add(Dense(128,input_dim=2,activation='relu')) #

model.add(Dense(1,activation='sigmoid'))
#3 fit

model.compile(loss= 'binary_crossentropy',optimizer='adam',metrics=['acc'])
hist=model.fit(x_data,y_data,batch_size=1,epochs=100)
#4 평가

y_predict = model.predict(x_data)
print(x_data,' 의 예측값 : ',y_predict)

results = model.evaluate(x_data, y_data)
print('score : ',results[1]) # 0 : loss 1: acc

acc = hist.history['acc']
print('acc : ',acc[-1])

#print('acc : ',acc.argmax())  argmax 사용해서 수정
'''
[[0, 0], [0, 1], [1, 0], [1, 1]]  의 예측값 :  [[0.47957724]
 [0.5828841 ]
 [0.59466934]
 [0.6899024 ]]
1/1 [==============================] - 0s 88ms/step - loss: 0.7209 - acc: 0.7500
score :  0.7208746671676636
acc :  0.75
relu사용
[[0, 0], [0, 1], [1, 0], [1, 1]]  의 예측값 :  [[0.46601525]
 [0.7628212 ]
 [0.75534844]
 [0.3418606 ]]
1/1 [==============================] - 0s 94ms/step - loss: 0.3993 - acc: 1.0000
score :  0.3992585837841034
acc :  1.0
'''