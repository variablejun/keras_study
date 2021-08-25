
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
#1데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],[1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.5, 1.4, 1.3]])

# print(x.shape)  배열 구조 확인  2,10

x = np.transpose(x) # 행 열 전환

# print(x.shape)  10,2
# reshape 해주여야한다 왜냐하면 열은 곧 속성을 나타내기 때문이다
# input레이어 구성시 열 우선 행 무시 열의 개수는 곧 input디멘션의 개수가 된다. 

y = np.array([11,12,13,14,15,16,17,18,19,20])
#  x 와 y의 행의 개수는 같아야한다 
# print(y.shape)  10,
# 모델구성
model = Sequential()
model.add(Dense(10, input_dim=2))
model.add(Dense(5))
model.add(Dense(10))
model.add(Dense(4))
model.add(Dense(10))
model.add(Dense(3))
model.add(Dense(10))

model.add(Dense(1))
# 컴파일 훈련
model.compile(loss="mse",optimizer="adam")
model.fit(x,y,epochs=100,batch_size=1)

# 평가 예측 
loss = model.evaluate(x,y)
print('loss : ', loss)

x_pred = np.array([[10, 1.3]])
result = model.predict(x_pred)

plt.scatter(x[:,0], y)
plt.scatter(x[:,1], y)
plt.plot(x[:,0],result,color='red')
plt.plot(x[:,1],result,color='blue')
plt.show()

'''

print('10과 1.3의 예측값 : ', result)
model.fit(x,y,epochs=895,batch_size=1)
loss :  3.841110810753889e-06
10과 1.3의 예측값 :  [[19.999708]]

[1,2,3] (3, )
[[1,2,3]] 1행 3열
[[1,2],[3,4],[5,6]] 3행 2열
[[[1,2,3],[4,5,6]]]  1,2,3
[[[[1,2],[3,4],[5,6]]] 1,3,2
[[[1],[2]],[[3],[4]]]  2,2,1

컬럼 = 피처 = 열 = 특성

데이터를 묶는 마지막 괄호는 계산할때 제외한다
'''

