from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import accuracy_score

#1 data
x_data = [[0,0],[0,1],[1,0],[1,1]]

y_data = [0,1,1,1]

#2 model

model = LinearSVC()

#3 fit

model.fit(x_data,y_data)

#4 평가

y_predict = model.predict(x_data)
print(x_data,' 의 예측값 : ',y_predict)

results= model.score(x_data, y_data)
print('score : ',results)

acc = accuracy_score(y_data,y_predict)
print('acc : ',acc)
'''
[[0, 0], [0, 1], [1, 0], [1, 1]]  의 예측값 :  [0 1 1 1]
score :  1.0
acc :  1.0
'''