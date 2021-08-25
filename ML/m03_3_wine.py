from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
dataset = load_wine()

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape) # (569, 30) (569,) y 0 과 1로 구성

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = LinearSVC()

model.fit(x_train, y_train)

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
y_predic :  [2 1 1 0 1 1 2 0 0]

SVC
acc :  1.0
results :  1.0
y_predic :  [2 1 1 0 1 1 2 0 0]

KNeighborsClassifier
acc :  1.0
results :  1.0
y_predic :  [2 1 1 0 1 1 2 0 0]

KNeighborsRegressor
acc :  1.0
results :  1.0
y_predic :  [2. 1. 1. 0. 1. 1. 2. 0. 0.]

LogisticRegression
acc :  1.0
results :  1.0
y_predic :  [1 1 1 1 1 0 0 1 1 1 0 1 1 0 1 1 0 1 0 0 1 0 1 0 1 1 0 1 1]

LinearRegression
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

DecisionTreeClassifier
acc :  1.0
results :  1.0
y_predic :  [2 1 1 0 1 1 2 0 0]

DecisionTreeRegressor
acc :  1.0
results :  1.0
y_predic :  [2. 1. 1. 0. 1. 1. 2. 0. 0.]

RandomForestClassifier
acc :  1.0
results :  1.0
y_predic :  [2 1 1 0 1 1 2 0 0]

RandomForestRegressor
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
'''