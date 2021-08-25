import numpy as np 
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=76)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore') # 워닝 무시

allAllalgorythsm1 = all_estimators(type_filter='classifier') # 분류 모델 모델의 개수는  41
allAllalgorythsm2 = all_estimators(type_filter='regressor') # 회귀모델 모델의 개수는  54

print(allAllalgorythsm1)
print(allAllalgorythsm2)
print('모델의 개수는 ',len(allAllalgorythsm2))  
for (name, algorythsm) in allAllalgorythsm1:
     try :
          model = algorythsm()
          model.fit(x_train,y_train)
          y_predict = model.predict(x_test)
          acc = accuracy_score(y_test,y_predict)
          print(name,'의 acc : ',acc)
     except:
          print(name,'없는것')


'''
MinMaxScaler
AdaBoostClassifier 의 acc :  0.9655172413793104
BaggingClassifier 의 acc :  0.9310344827586207
BernoulliNB 의 acc :  0.6896551724137931
CalibratedClassifierCV 의 acc :  1.0
CategoricalNB 의 acc :  0.6896551724137931
ClassifierChain 없는것
ComplementNB 의 acc :  0.896551724137931
DecisionTreeClassifier 의 acc :  0.9310344827586207
DummyClassifier 의 acc :  0.6896551724137931
ExtraTreeClassifier 의 acc :  0.9310344827586207
ExtraTreesClassifier 의 acc :  1.0
GaussianNB 의 acc :  0.9655172413793104
GaussianProcessClassifier 의 acc :  1.0
GradientBoostingClassifier 의 acc :  0.9655172413793104
HistGradientBoostingClassifier 의 acc :  0.9655172413793104
KNeighborsClassifier 의 acc :  0.9655172413793104
LabelPropagation 의 acc :  1.0
LabelSpreading 의 acc :  1.0
LinearDiscriminantAnalysis 의 acc :  0.9655172413793104
LinearSVC 의 acc :  1.0
LogisticRegression 의 acc :  1.0
LogisticRegressionCV 의 acc :  1.0
MLPClassifier 의 acc :  1.0
MultiOutputClassifier 없는것
MultinomialNB 의 acc :  0.8275862068965517
NearestCentroid 의 acc :  0.9655172413793104
NuSVC 의 acc :  0.9655172413793104
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  0.9655172413793104
Perceptron 의 acc :  0.9655172413793104
QuadraticDiscriminantAnalysis 의 acc :  0.9655172413793104
RadiusNeighborsClassifier 의 acc :  0.8620689655172413
RandomForestClassifier 의 acc :  0.9655172413793104
RidgeClassifier 의 acc :  0.9655172413793104
RidgeClassifierCV 의 acc :  0.9655172413793104
SGDClassifier 의 acc :  1.0
SVC 의 acc :  1.0
StackingClassifier 없는것
VotingClassifier 없는것
===============================================================
StandardScaler
AdaBoostClassifier 의 acc :  0.9655172413793104
BaggingClassifier 의 acc :  1.0
BernoulliNB 의 acc :  0.9655172413793104
CalibratedClassifierCV 의 acc :  0.9655172413793104
CategoricalNB 없는것
ClassifierChain 없는것
ComplementNB 없는것
DecisionTreeClassifier 의 acc :  0.9310344827586207
DummyClassifier 의 acc :  0.6896551724137931
ExtraTreeClassifier 의 acc :  0.9655172413793104
ExtraTreesClassifier 의 acc :  1.0
GaussianNB 의 acc :  0.9655172413793104
GaussianProcessClassifier 의 acc :  1.0
GradientBoostingClassifier 의 acc :  0.9655172413793104
HistGradientBoostingClassifier 의 acc :  0.9655172413793104
KNeighborsClassifier 의 acc :  0.9655172413793104
LabelPropagation 의 acc :  1.0
LabelSpreading 의 acc :  1.0
LinearDiscriminantAnalysis 의 acc :  0.9655172413793104
LinearSVC 의 acc :  1.0
LogisticRegression 의 acc :  1.0
LogisticRegressionCV 의 acc :  1.0
MLPClassifier 의 acc :  1.0
MultiOutputClassifier 없는것
MultinomialNB 없는것
NearestCentroid 의 acc :  0.9655172413793104
NuSVC 의 acc :  0.9655172413793104
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  1.0
Perceptron 의 acc :  1.0
QuadraticDiscriminantAnalysis 의 acc :  0.9655172413793104
RadiusNeighborsClassifier 없는것
RandomForestClassifier 의 acc :  1.0
RidgeClassifier 의 acc :  0.9655172413793104
RidgeClassifierCV 의 acc :  0.9655172413793104
SGDClassifier 의 acc :  1.0
SVC 의 acc :  1.0
StackingClassifier 없는것
VotingClassifier 없는것
'''