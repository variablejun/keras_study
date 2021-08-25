import numpy as np 
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target
from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import r2_score,accuracy_score
from sklearn.utils import all_estimators

import warnings
warnings.filterwarnings('ignore') # 워닝 무시

allAllalgorythsm1 = all_estimators(type_filter='classifier') # 분류 모델 모델의 개수는  41
allAllalgorythsm2 = all_estimators(type_filter='regressor') # 회귀모델 모델의 개수는  54

print(allAllalgorythsm1)
print(allAllalgorythsm2)
print('모델의 개수는 ',len(allAllalgorythsm2))  
kfold = KFold(n_splits=5,random_state=66,shuffle=True)
for (name, algorythsm) in allAllalgorythsm1:
     try :
          model = algorythsm()
          score = cross_val_score(model,x,y,cv=kfold)
          print(name,'의 acc : ',score) # tap 한칸 shift + tap 한칸뒤로
          print("평균 : ",round(np.mean(score),4))
     except:
          print(name,'없는것')
#여러가지 경우에 따라서 파라미터가 안맞거나 해서 fit을 못할경우 따로 빼서 이름만 출력

     
'''
AdaBoostClassifier 의 acc :  [0.94736842 0.99122807 0.94736842 0.96491228 0.97345133]
평균 :  0.9649
BaggingClassifier 의 acc :  [0.95614035 0.92982456 0.93859649 0.93859649 0.94690265]
평균 :  0.942
BernoulliNB 의 acc :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
평균 :  0.6274
CalibratedClassifierCV 의 acc :  [0.89473684 0.93859649 0.89473684 0.92982456 0.97345133]
평균 :  0.9263
CategoricalNB 의 acc :  [nan nan nan nan nan]
평균 :  nan
ClassifierChain 없는것
ComplementNB 의 acc :  [0.86842105 0.92982456 0.87719298 0.9122807  0.89380531]
평균 :  0.8963
DecisionTreeClassifier 의 acc :  [0.93859649 0.92105263 0.92982456 0.87719298 0.9380531 ]
평균 :  0.9209
DummyClassifier 의 acc :  [0.64035088 0.65789474 0.62280702 0.5877193  0.62831858]
평균 :  0.6274
ExtraTreeClassifier 의 acc :  [0.86842105 0.95614035 0.87719298 0.92105263 0.95575221]
평균 :  0.9157
ExtraTreesClassifier 의 acc :  [0.96491228 0.97368421 0.96491228 0.94736842 0.99115044]
평균 :  0.9684
GaussianNB 의 acc :  [0.93859649 0.96491228 0.9122807  0.93859649 0.95575221]
평균 :  0.942
GaussianProcessClassifier 의 acc :  [0.87719298 0.89473684 0.89473684 0.94736842 0.94690265]
평균 :  0.9122
GradientBoostingClassifier 의 acc :  [0.95614035 0.96491228 0.95614035 0.93859649 0.98230088]
평균 :  0.9596
HistGradientBoostingClassifier 의 acc :  [0.97368421 0.98245614 0.96491228 0.96491228 0.98230088]
평균 :  0.9737
KNeighborsClassifier 의 acc :  [0.92105263 0.92105263 0.92105263 0.92105263 0.95575221]
평균 :  0.928
LabelPropagation 의 acc :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
평균 :  0.3902
LabelSpreading 의 acc :  [0.36842105 0.35964912 0.4122807  0.42105263 0.38938053]
평균 :  0.3902
LinearDiscriminantAnalysis 의 acc :  [0.94736842 0.98245614 0.94736842 0.95614035 0.97345133]
평균 :  0.9614
LinearSVC 의 acc :  [0.92982456 0.85087719 0.92105263 0.92105263 0.91150442]
평균 :  0.9069
LogisticRegression 의 acc :  [0.93859649 0.95614035 0.88596491 0.94736842 0.96460177]
평균 :  0.9385
LogisticRegressionCV 의 acc :  [0.96491228 0.97368421 0.92105263 0.96491228 0.96460177]
평균 :  0.9578
MLPClassifier 의 acc :  [0.90350877 0.92982456 0.9122807  0.93859649 0.96460177]
평균 :  0.9298
MultiOutputClassifier 없는것
MultinomialNB 의 acc :  [0.85964912 0.92105263 0.87719298 0.9122807  0.89380531]
평균 :  0.8928
NearestCentroid 의 acc :  [0.86842105 0.89473684 0.85964912 0.9122807  0.91150442]
평균 :  0.8893
NuSVC 의 acc :  [0.85964912 0.9122807  0.83333333 0.87719298 0.88495575]
평균 :  0.8735
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  [0.8245614  0.92982456 0.83333333 0.57017544 0.94690265]
평균 :  0.821
Perceptron 의 acc :  [0.40350877 0.80701754 0.85964912 0.86842105 0.94690265]
평균 :  0.7771
QuadraticDiscriminantAnalysis 의 acc :  [0.93859649 0.95614035 0.93859649 0.98245614 0.94690265]
평균 :  0.9525
RadiusNeighborsClassifier 의 acc :  [nan nan nan nan nan]
평균 :  nan
RandomForestClassifier 의 acc :  [0.97368421 0.96491228 0.96491228 0.95614035 0.98230088]
평균 :  0.9684
RidgeClassifier 의 acc :  [0.95614035 0.98245614 0.92105263 0.95614035 0.95575221]
평균 :  0.9543
RidgeClassifierCV 의 acc :  [0.94736842 0.97368421 0.93859649 0.95614035 0.96460177]
평균 :  0.9561
SGDClassifier 의 acc :  [0.69298246 0.66666667 0.83333333 0.90350877 0.9380531 ]
평균 :  0.8069
SVC 의 acc :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177]
평균 :  0.921
StackingClassifier 없는것
VotingClassifier 없는것
'''


