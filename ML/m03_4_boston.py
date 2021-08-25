
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 

from sklearn.datasets import load_boston
datasets = load_boston()

#1.data
x = datasets.data
y = datasets.target

print(np.max(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

#scaler = MinMaxScaler()
scaler = PowerTransformer()
scaler.fit(x_train)  
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test) 



#2.모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = RandomForestRegressor()

#3.complie/훈련

model.fit(x_train, y_train) 

from sklearn.metrics import r2_score,accuracy_score 
y_predic = model.predict(x_test)
acc = accuracy_score(y_test,y_predic) 
print('acc : ',acc)
results = model.score(x_test, y_test)
print('results : ',results)
y_predic2 = model.predict(x_test[:5])
print('y_predic : ',y_predic)


'''
LinearSVC
ValueError: Unknown label type: 'continuous'

SVC
ValueError: Unknown label type: 'continuous'

KNeighborsClassifier
ValueError: Unknown label type: 'continuous'

KNeighborsRegressor
ValueError: continuous is not supported

LogisticRegression
ValueError: Unknown label type: 'continuous'

LinearRegression
ValueError: continuous is not supported

DecisionTreeClassifier
ValueError: Unknown label type: 'continuous'

DecisionTreeRegressor
ValueError: continuous is not supported

RandomForestClassifier
ValueError: Unknown label type: 'continuous'

RandomForestRegressor
ValueError: continuous is not supported

'''