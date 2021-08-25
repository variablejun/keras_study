import numpy as np 
from sklearn.datasets import load_boston

dataset = load_boston()
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
for (name, algorythsm) in allAllalgorythsm2:
     try :
          model = algorythsm()
          score = cross_val_score(model,x,y,cv=kfold)
          print(name,'의 score : ',score) # tap 한칸 shift + tap 한칸뒤로
          print("평균 : ",round(np.mean(score),4))
     except:
          print(name,'없는것')
#여러가지 경우에 따라서 파라미터가 안맞거나 해서 fit을 못할경우 따로 빼서 이름만 출력

     
'''
ARDRegression 의 score :  [0.80125693 0.76317071 0.56809285 0.6400258  0.71991866]
평균 :  0.6985
AdaBoostRegressor 의 score :  [0.90590669 0.79821339 0.80027451 0.83140451 0.87164967]
평균 :  0.8415
BaggingRegressor 의 score :  [0.88891814 0.79647157 0.82612568 0.84590737 0.88897669]
평균 :  0.8493
BayesianRidge 의 score :  [0.79379186 0.81123808 0.57943979 0.62721388 0.70719051]
평균 :  0.7038
CCA 의 score :  [0.79134772 0.73828469 0.39419624 0.5795108  0.73224276]
평균 :  0.6471
DecisionTreeRegressor 의 score :  [0.78990224 0.60273714 0.81415688 0.7336607  0.83054886]
평균 :  0.7542
DummyRegressor 의 score :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
평균 :  -0.0135
ElasticNet 의 score :  [0.73383355 0.76745241 0.59979782 0.60616114 0.64658354]
평균 :  0.6708
ElasticNetCV 의 score :  [0.71677604 0.75276545 0.59116613 0.59289916 0.62888608]
평균 :  0.6565
ExtraTreeRegressor 의 score :  [0.84731028 0.7083747  0.60086001 0.72161571 0.72243424]
평균 :  0.7201
ExtraTreesRegressor 의 score :  [0.9320587  0.8594843  0.78020783 0.88069515 0.92910021]
평균 :  0.8763
GammaRegressor 의 score :  [-0.00058757 -0.03146716 -0.00463664 -0.02807276 -0.00298635]
평균 :  -0.0136
GaussianProcessRegressor 의 score :  [-6.07310526 -5.51957093 -6.33482574 -6.36383476 -5.35160828]
평균 :  -5.9286
GradientBoostingRegressor 의 score :  [0.94578359 0.83719345 0.82599792 0.88576808 0.93119582]
평균 :  0.8852
HistGradientBoostingRegressor 의 score :  [0.93235978 0.82415907 0.78740524 0.88879806 0.85766226]
평균 :  0.8581
HuberRegressor 의 score :  [0.74400323 0.64244715 0.52848946 0.37100122 0.63403398]
평균 :  0.584
IsotonicRegression 의 score :  [nan nan nan nan nan]
평균 :  nan
KNeighborsRegressor 의 score :  [0.59008727 0.68112533 0.55680192 0.4032667  0.41180856]
평균 :  0.5286
KernelRidge 의 score :  [0.83333255 0.76712443 0.5304997  0.5836223  0.71226555]
평균 :  0.6854
Lars 의 score :  [0.77467361 0.79839316 0.5903683  0.64083802 0.68439384]
평균 :  0.6977
LarsCV 의 score :  [0.80141197 0.77573678 0.57807429 0.60068407 0.70833854]
평균 :  0.6928
Lasso 의 score :  [0.7240751  0.76027388 0.60141929 0.60458689 0.63793473]
평균 :  0.6657
LassoCV 의 score :  [0.71314939 0.79141061 0.60734295 0.61617714 0.66137127]
평균 :  0.6779
LassoLars 의 score :  [-0.00053702 -0.03356375 -0.00476023 -0.02593069 -0.00275911]
평균 :  -0.0135
LassoLarsCV 의 score :  [0.80301044 0.77573678 0.57807429 0.60068407 0.72486787]
평균 :  0.6965
LassoLarsIC 의 score :  [0.81314239 0.79765276 0.59012698 0.63974189 0.72415009]
평균 :  0.713
LinearRegression 의 score :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
평균 :  0.7128
LinearSVR 의 score :  [0.74434274 0.68470565 0.38573342 0.38017436 0.62564968]
평균 :  0.5641
MLPRegressor 의 score :  [0.48166997 0.535649   0.32418468 0.43133888 0.57983738]
평균 :  0.4705
MultiOutputRegressor 없는것
MultiTaskElasticNet 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskElasticNetCV 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskLasso 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskLassoCV 의 score :  [nan nan nan nan nan]
평균 :  nan
NuSVR 의 score :  [0.2594254  0.33427351 0.263857   0.11914968 0.170599  ]
평균 :  0.2295
OrthogonalMatchingPursuit 의 score :  [0.58276176 0.565867   0.48689774 0.51545117 0.52049576]
평균 :  0.5343
OrthogonalMatchingPursuitCV 의 score :  [0.75264599 0.75091171 0.52333619 0.59442374 0.66783377]
평균 :  0.6578
PLSCanonical 의 score :  [-2.23170797 -2.33245351 -2.89155602 -2.14746527 -1.44488868]
평균 :  -2.2096
PLSRegression 의 score :  [0.80273131 0.76619347 0.52249555 0.59721829 0.73503313]
평균 :  0.6847
PassiveAggressiveRegressor 의 score :  [ 0.23957555 -1.6548686   0.26394796  0.25034992  0.18877643]
평균 :  -0.1424
PoissonRegressor 의 score :  [0.85659255 0.8189989  0.66691488 0.67998192 0.75195656]
평균 :  0.7549
RANSACRegressor 의 score :  [0.70141872 0.01748121 0.4160419  0.43132836 0.4201483 ]
평균 :  0.3973
RadiusNeighborsRegressor 의 score :  [nan nan nan nan nan]
평균 :  nan
RandomForestRegressor 의 score :  [0.91946749 0.86488394 0.82239246 0.8820721  0.90663276]
평균 :  0.8791
RegressorChain 없는것
Ridge 의 score :  [0.80984876 0.80618063 0.58111378 0.63459427 0.72264776]
평균 :  0.7109
RidgeCV 의 score :  [0.81125292 0.80010535 0.58888304 0.64008984 0.72362912]
평균 :  0.7128
SGDRegressor 의 score :  [-4.24843897e+24 -6.16775067e+25 -6.52509087e+26 -3.89793175e+26
 -1.09130680e+26]
평균 :  -2.4347177747268017e+26
SVR 의 score :  [0.23475113 0.31583258 0.24121157 0.04946335 0.14020554]
평균 :  0.1963
StackingRegressor 없는것
TheilSenRegressor 의 score :  [0.79606135 0.74114823 0.58904157 0.54256428 0.7200797 ]
평균 :  0.6778
TransformedTargetRegressor 의 score :  [0.81112887 0.79839316 0.59033016 0.64083802 0.72332215]
평균 :  0.7128
TweedieRegressor 의 score :  [0.7492543  0.75457294 0.56286929 0.57989884 0.63242475]
평균 :  0.6558
VotingRegressor 없는것
'''


