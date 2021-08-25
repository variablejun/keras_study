import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)
'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
머신러닝에선 원핫인코딩 적용안해도 해줌
ValueError: y should be a 1d array, got an array of shape (142, 3) instead.
1차원을 줘야하는데 2차원을줫음 원핫인코딩 때문
대부분의 머신러닝들을 y를 1차원으로 받아들여서 오류가나고 따로 안해줘도됨
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=76)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것
model = RandomForestClassifier()

model.fit(x_train,y_train)
from sklearn.metrics import r2_score,accuracy_score # r2 회기모델 평가  acc 분류모델평가
y_predic = model.predict(x_test)
acc = accuracy_score(y_test,y_predic) # 딥러닝에서 acc와 결과가 같다 딥러닝evaluate에서 xtest를 predic해서 ytest와 비교하기때문이다.
print('acc : ',acc)
results = model.score(x_test, y_test)
print('results : ',results)
y_predic2 = model.predict(x_test[:5])
print('y_predic : ',y_predic)
'''
LinearSVC
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]

SVC
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]

KNeighborsClassifier
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]

LogisticRegression
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]

DecisionTreeClassifier
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]

RandomForestClassifier
acc :  1.0
results :  1.0
y_predic :  [0 1 0 2 2 1 0 1]


'''