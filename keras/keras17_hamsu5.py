import numpy as np

#  데이터구성
x = np.array([range(100),range(301,401),range(1,101),range(100),range(401,501)])
x = np.transpose(x)
y = np.array([range(711,811),range(101,201)])
y = np.transpose(y)
print(x.shape , y.shape) # 100,5  100,2

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
# 단순하게 시퀀셜로 내려가는 경우 input,output제외하고 레이어명을 다르게 설정해도됨 Dense 안에 name으로 이름줄 수도있다
input1 = Input(shape=(5,))
xx = Dense(3)(input1)
xx = Dense(4)(xx)
xx = Dense(10)(xx)
output1 = Dense(2)(xx)

model = Model(inputs = input1, outputs = output1) # 모델 여러개를 합치거나 순서를 바꾸기 쉽다.
# 모델을 명시하는 위치가 시퀀셜형과 다르게 명시해준다
# (inputshape + b) * outputshape
model.summary()