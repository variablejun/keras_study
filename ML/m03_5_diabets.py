from sklearn.datasets import load_diabetes
import numpy as np
from re import M
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
# import pandas as pd

#1.data
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x = (x - np.min(x)) / (np.max(x) - np.min(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=66)

print(np.shape(x), np.shape(y))
print(datasets.feature_names)
#['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']


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
acc :  0.0
results :  0.0

SVC
acc :  0.0
results :  0.0

KNeighborsClassifier
acc :  0.0
results :  0.0

KNeighborsRegressor
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

LogisticRegression
acc :  0.0
results :  0.0

LinearRegression
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets

DecisionTreeClassifier
acc :  0.007518796992481203
results :  0.007518796992481203

DecisionTreeRegressor
acc :  0.007518796992481203
results :  -0.039311138053854444

RandomForestClassifier
acc :  0.015037593984962405
results :  0.015037593984962405

RandomForestRegressor
ValueError: Classification metrics can't handle a mix of multiclass and continuous targets
'''