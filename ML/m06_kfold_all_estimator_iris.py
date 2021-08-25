import numpy as np 
from sklearn.datasets import load_iris

dataset = load_iris()
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
AdaBoostClassifier 의 acc :  [0.63333333 0.93333333 1.         0.9        0.96666667]
평균 :  0.8867
BaggingClassifier 의 acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
평균 :  0.96
BernoulliNB 의 acc :  [0.3        0.33333333 0.3        0.23333333 0.3       ]
평균 :  0.2933
CalibratedClassifierCV 의 acc :  [0.9        0.83333333 1.         0.86666667 0.96666667]
평균 :  0.9133
CategoricalNB 의 acc :  [0.9        0.93333333 0.93333333 0.9        1.        ]
평균 :  0.9333
ClassifierChain 없는것
ComplementNB 의 acc :  [0.66666667 0.66666667 0.7        0.6        0.7       ]
평균 :  0.6667
DecisionTreeClassifier 의 acc :  [0.96666667 0.96666667 1.         0.9        0.93333333]
평균 :  0.9533
DummyClassifier 의 acc :  [0.3        0.33333333 0.3        0.23333333 0.3       ]
평균 :  0.2933
ExtraTreeClassifier 의 acc :  [0.93333333 1.         0.96666667 0.9        0.96666667]
평균 :  0.9533
ExtraTreesClassifier 의 acc :  [0.96666667 0.96666667 1.         0.86666667 0.96666667]
평균 :  0.9533
GaussianNB 의 acc :  [0.96666667 0.9        1.         0.9        0.96666667]
평균 :  0.9467
GaussianProcessClassifier 의 acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
평균 :  0.96
GradientBoostingClassifier 의 acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
평균 :  0.9667
HistGradientBoostingClassifier 의 acc :  [0.86666667 0.96666667 1.         0.9        0.96666667]
평균 :  0.94
KNeighborsClassifier 의 acc :  [0.96666667 0.96666667 1.         0.9        0.96666667]
평균 :  0.96
LabelPropagation 의 acc :  [0.93333333 1.         1.         0.9        0.96666667]
평균 :  0.96
LabelSpreading 의 acc :  [0.93333333 1.         1.         0.9        0.96666667]
평균 :  0.96
LinearDiscriminantAnalysis 의 acc :  [1.  1.  1.  0.9 1. ]
평균 :  0.98
LinearSVC 의 acc :  [0.96666667 0.96666667 1.         0.9        1.        ]
평균 :  0.9667
LogisticRegression 의 acc :  [1.         0.96666667 1.         0.9        0.96666667]
평균 :  0.9667
LogisticRegressionCV 의 acc :  [1.         0.96666667 1.         0.9        1.        ]
평균 :  0.9733
MLPClassifier 의 acc :  [1.         0.96666667 1.         0.93333333 1.        ]
평균 :  0.98
MultiOutputClassifier 없는것
MultinomialNB 의 acc :  [0.96666667 0.93333333 1.         0.93333333 1.        ]
평균 :  0.9667
NearestCentroid 의 acc :  [0.93333333 0.9        0.96666667 0.9        0.96666667]
평균 :  0.9333
NuSVC 의 acc :  [0.96666667 0.96666667 1.         0.93333333 1.        ]
평균 :  0.9733
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  [0.73333333 0.76666667 0.8        0.7        0.83333333]
평균 :  0.7667
Perceptron 의 acc :  [0.66666667 0.66666667 0.93333333 0.73333333 0.9       ]
평균 :  0.78
QuadraticDiscriminantAnalysis 의 acc :  [1.         0.96666667 1.         0.93333333 1.        ]
평균 :  0.98
RadiusNeighborsClassifier 의 acc :  [0.96666667 0.9        0.96666667 0.93333333 1.        ]
평균 :  0.9533
RandomForestClassifier 의 acc :  [0.93333333 0.96666667 1.         0.9        0.96666667]
평균 :  0.9533
RidgeClassifier 의 acc :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
평균 :  0.84
RidgeClassifierCV 의 acc :  [0.86666667 0.8        0.93333333 0.7        0.9       ]
평균 :  0.84
SGDClassifier 의 acc :  [0.93333333 0.83333333 0.73333333 0.76666667 0.86666667]
평균 :  0.8267
SVC 의 acc :  [0.96666667 0.96666667 1.         0.93333333 0.96666667]
평균 :  0.9667
StackingClassifier 없는것
VotingClassifier 없는것
'''


