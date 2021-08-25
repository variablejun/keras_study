import numpy as np 
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
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
ARDRegression 의 score :  [0.49874835 0.48765748 0.56284846 0.37728801 0.53474369]
평균 :  0.4923
AdaBoostRegressor 의 score :  [0.34321915 0.45971876 0.48749887 0.37699194 0.45956748]
평균 :  0.4254
BaggingRegressor 의 score :  [0.37524355 0.42172138 0.38086246 0.33564274 0.40783732]
평균 :  0.3843
BayesianRidge 의 score :  [0.50082189 0.48431051 0.55459312 0.37600508 0.5307344 ]
평균 :  0.4893
CCA 의 score :  [0.48696409 0.42605855 0.55244322 0.21708682 0.50764701]
평균 :  0.438
DecisionTreeRegressor 의 score :  [-0.26200457 -0.12897221 -0.13019926 -0.08616484  0.12497562]
평균 :  -0.0965
DummyRegressor 의 score :  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03]
평균 :  -0.0033
ElasticNet 의 score :  [ 0.00810127  0.00637294  0.00924848  0.0040621  -0.00081988]
평균 :  0.0054
ElasticNetCV 의 score :  [0.43071558 0.461506   0.49133954 0.35674829 0.4567084 ]
평균 :  0.4394
ExtraTreeRegressor 의 score :  [-0.13267244  0.00093007 -0.09236731 -0.38451336  0.15626263]
평균 :  -0.0905
ExtraTreesRegressor 의 score :  [0.36424034 0.48051354 0.51101523 0.38877623 0.45493328]
평균 :  0.4399
GammaRegressor 의 score :  [ 0.00523561  0.00367973  0.0060814   0.00174734 -0.00306898]
평균 :  0.0027
GaussianProcessRegressor 의 score :  [ -5.6360757  -15.27401119  -9.94981439 -12.46884878 -12.04794389]
평균 :  -11.0753
GradientBoostingRegressor 의 score :  [0.39229176 0.48782958 0.48098403 0.39528846 0.44659167]
평균 :  0.4406
HistGradientBoostingRegressor 의 score :  [0.28899498 0.43812684 0.51713242 0.37267554 0.35643755]
평균 :  0.3947
HuberRegressor 의 score :  [0.50334705 0.47508237 0.54650576 0.36883712 0.5173073 ]
평균 :  0.4822
IsotonicRegression 의 score :  [nan nan nan nan nan]
평균 :  nan
KNeighborsRegressor 의 score :  [0.39683913 0.32569788 0.43311217 0.32635899 0.35466969]
평균 :  0.3673
KernelRidge 의 score :  [-3.38476443 -3.49366182 -4.0996205  -3.39039111 -3.60041537]
평균 :  -3.5938
Lars 의 score :  [ 0.49198665 -0.66475442 -1.04410299 -0.04236657  0.51190679]
평균 :  -0.1495
LarsCV 의 score :  [0.4931481  0.48774421 0.55427158 0.38001456 0.52413596]
평균 :  0.4879
Lasso 의 score :  [0.34315574 0.35348212 0.38594431 0.31614536 0.3604865 ]
평균 :  0.3518
LassoCV 의 score :  [0.49799859 0.48389346 0.55926851 0.37740074 0.51636393]
평균 :  0.487
LassoLars 의 score :  [0.36543887 0.37812653 0.40638095 0.33639271 0.38444891]
평균 :  0.3742
LassoLarsCV 의 score :  [0.49719648 0.48426377 0.55975856 0.37984022 0.51190679]
평균 :  0.4866
LassoLarsIC 의 score :  [0.49940515 0.49108789 0.56130589 0.37942384 0.5247894 ]
평균 :  0.4912
LinearRegression 의 score :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
평균 :  0.4876
LinearSVR 의 score :  [-0.33470258 -0.31657282 -0.41638485 -0.30277364 -0.47360934]
평균 :  -0.3688
MLPRegressor 의 score :  [-2.79422923 -3.18288256 -3.27584551 -2.88947374 -3.03112597]
평균 :  -3.0347
MultiOutputRegressor 없는것
MultiTaskElasticNet 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskElasticNetCV 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskLasso 의 score :  [nan nan nan nan nan]
평균 :  nan
MultiTaskLassoCV 의 score :  [nan nan nan nan nan]
평균 :  nan
NuSVR 의 score :  [0.14471275 0.17351835 0.18539957 0.13894135 0.1663745 ]
평균 :  0.1618
OrthogonalMatchingPursuit 의 score :  [0.32934491 0.285747   0.38943221 0.19671679 0.35916077]
평균 :  0.3121
OrthogonalMatchingPursuitCV 의 score :  [0.47845357 0.48661326 0.55695148 0.37039612 0.53615516]
평균 :  0.4857
PLSCanonical 의 score :  [-0.97507923 -1.68534502 -0.8821301  -1.33987816 -1.16041996]
평균 :  -1.2086
PLSRegression 의 score :  [0.47661395 0.4762657  0.5388494  0.38191443 0.54717873]
평균 :  0.4842
PassiveAggressiveRegressor 의 score :  [0.43303352 0.48144778 0.5073907  0.36171009 0.44640923]
평균 :  0.446
PoissonRegressor 의 score :  [0.32061441 0.35803358 0.3666005  0.28203414 0.34340626]
평균 :  0.3341
RANSACRegressor 의 score :  [-0.26255076  0.29147661  0.08687522  0.1033501   0.42883212]
평균 :  0.1296
RadiusNeighborsRegressor 의 score :  [-1.54258856e-04 -2.98519672e-03 -1.53442062e-05 -3.80334913e-03
 -9.58335111e-03]
평균 :  -0.0033
RandomForestRegressor 의 score :  [0.36041953 0.4866929  0.47170212 0.41241244 0.43565825]
평균 :  0.4334
RegressorChain 없는것
Ridge 의 score :  [0.40936669 0.44788406 0.47057299 0.34467674 0.43339091]
평균 :  0.4212
RidgeCV 의 score :  [0.49525464 0.48761091 0.55171354 0.3801769  0.52749194]
평균 :  0.4884
SGDRegressor 의 score :  [0.3933391  0.44179008 0.46461185 0.32976867 0.41499158]
평균 :  0.4089
SVR 의 score :  [0.14331635 0.18438697 0.17864042 0.1424597  0.1468719 ]
평균 :  0.1591
StackingRegressor 없는것
TheilSenRegressor 의 score :  [0.50619565 0.46012546 0.55796475 0.33545401 0.54046537]
평균 :  0.48
TransformedTargetRegressor 의 score :  [0.50638911 0.48684632 0.55366898 0.3794262  0.51190679]
평균 :  0.4876
TweedieRegressor 의 score :  [ 0.00585525  0.00425899  0.00702558  0.00183408 -0.00315042]
평균 :  0.0032
VotingRegressor 없는것
'''


