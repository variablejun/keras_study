import numpy as np 
from sklearn.datasets import load_wine

dataset = load_wine()
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
AdaBoostClassifier 의 acc :  [0.88888889 0.86111111 0.88888889 0.94285714 0.97142857]
평균 :  0.9106
BaggingClassifier 의 acc :  [1.         0.91666667 0.88888889 0.94285714 0.97142857]
평균 :  0.944
BernoulliNB 의 acc :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
평균 :  0.399
CalibratedClassifierCV 의 acc :  [0.94444444 0.94444444 0.88888889 0.88571429 0.91428571]
평균 :  0.9156
CategoricalNB 의 acc :  [       nan        nan        nan 0.94285714        nan]
평균 :  nan
ClassifierChain 없는것
ComplementNB 의 acc :  [0.69444444 0.80555556 0.55555556 0.6        0.6       ]
평균 :  0.6511
DecisionTreeClassifier 의 acc :  [0.91666667 0.97222222 0.91666667 0.85714286 0.91428571]
평균 :  0.9154
DummyClassifier 의 acc :  [0.41666667 0.47222222 0.27777778 0.48571429 0.34285714]
평균 :  0.399
ExtraTreeClassifier 의 acc :  [0.91666667 0.97222222 0.75       0.94285714 0.82857143]
평균 :  0.8821
ExtraTreesClassifier 의 acc :  [1.         0.97222222 1.         0.97142857 1.        ]
평균 :  0.9887
GaussianNB 의 acc :  [1.         0.91666667 0.97222222 0.97142857 1.        ]
평균 :  0.9721
GaussianProcessClassifier 의 acc :  [0.44444444 0.30555556 0.55555556 0.62857143 0.45714286]
평균 :  0.4783
GradientBoostingClassifier 의 acc :  [0.97222222 0.91666667 0.88888889 0.97142857 0.97142857]
평균 :  0.9441
HistGradientBoostingClassifier 의 acc :  [0.97222222 0.94444444 1.         0.97142857 1.        ]
평균 :  0.9776
KNeighborsClassifier 의 acc :  [0.69444444 0.77777778 0.61111111 0.62857143 0.74285714]
평균 :  0.691
LabelPropagation 의 acc :  [0.52777778 0.47222222 0.5        0.4        0.54285714]
평균 :  0.4886
LabelSpreading 의 acc :  [0.52777778 0.47222222 0.5        0.4        0.54285714]
평균 :  0.4886
LinearDiscriminantAnalysis 의 acc :  [1.         0.97222222 1.         0.97142857 1.        ]
평균 :  0.9887
LinearSVC 의 acc :  [0.55555556 0.88888889 0.63888889 0.8        0.8       ]
평균 :  0.7367
LogisticRegression 의 acc :  [0.97222222 0.94444444 0.94444444 0.94285714 1.        ]
평균 :  0.9608
LogisticRegressionCV 의 acc :  [1.         0.94444444 0.97222222 0.94285714 0.97142857]
평균 :  0.9662
MLPClassifier 의 acc :  [0.5        0.91666667 0.55555556 0.91428571 0.57142857]
평균 :  0.6916
MultiOutputClassifier 없는것
MultinomialNB 의 acc :  [0.77777778 0.91666667 0.86111111 0.82857143 0.82857143]
평균 :  0.8425
NearestCentroid 의 acc :  [0.69444444 0.72222222 0.69444444 0.77142857 0.74285714]
평균 :  0.7251
NuSVC 의 acc :  [0.91666667 0.86111111 0.91666667 0.85714286 0.8       ]
평균 :  0.8703
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  [0.63888889 0.77777778 0.55555556 0.31428571 0.57142857]
평균 :  0.5716
Perceptron 의 acc :  [0.61111111 0.80555556 0.47222222 0.48571429 0.62857143]
평균 :  0.6006
QuadraticDiscriminantAnalysis 의 acc :  [0.97222222 1.         1.         1.         1.        ]
평균 :  0.9944
RadiusNeighborsClassifier 의 acc :  [nan nan nan nan nan]
평균 :  nan
RandomForestClassifier 의 acc :  [1.         0.94444444 1.         0.97142857 1.        ]
평균 :  0.9832
RidgeClassifier 의 acc :  [1.         1.         1.         0.97142857 1.        ]
평균 :  0.9943
RidgeClassifierCV 의 acc :  [1.         1.         1.         0.97142857 1.        ]
평균 :  0.9943
SGDClassifier 의 acc :  [0.52777778 0.80555556 0.5        0.6        0.57142857]
평균 :  0.601
SVC 의 acc :  [0.69444444 0.69444444 0.61111111 0.62857143 0.6       ]
평균 :  0.6457
StackingClassifier 없는것
VotingClassifier 없는것
'''


