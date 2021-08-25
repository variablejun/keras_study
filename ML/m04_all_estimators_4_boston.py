import numpy as np 
from sklearn.datasets import load_boston

dataset = load_boston()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

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
for (name, algorythsm) in allAllalgorythsm2:
     try :
          model = algorythsm()
          model.fit(x_train,y_train)
          y_predict = model.predict(x_test)
          acc = r2_score(y_test,y_predict)
          print(name,'의 acc : ',acc)
     except:
          print(name,'없는것')


'''
분류모델로 돌릴경우 값도 제대로 안나오고 속도도 많이 느리다.
MinMaxScaler
ARDRegression 의 acc :  0.7207175586512667
AdaBoostRegressor 의 acc :  0.8295615566462063
BaggingRegressor 의 acc :  0.8575032894948996
BayesianRidge 의 acc :  0.721603290752266
CCA 의 acc :  0.6236875820293413
DecisionTreeRegressor 의 acc :  0.7464587906775318
DummyRegressor 의 acc :  -0.014319569088965922
ElasticNet 의 acc :  0.17682283998086357
ElasticNetCV 의 acc :  0.7308918525077632
ExtraTreeRegressor 의 acc :  0.6980896559037671
ExtraTreesRegressor 의 acc :  0.8885028103361561
GammaRegressor 의 acc :  0.19926394493329658
GaussianProcessRegressor 의 acc :  -0.2478397895482829
GradientBoostingRegressor 의 acc :  0.8729474608917726
HistGradientBoostingRegressor 의 acc :  0.8607348134767078
HuberRegressor 의 acc :  0.7464254354183177
IsotonicRegression 없는것
KNeighborsRegressor 의 acc :  0.7008122126350649
KernelRidge 의 acc :  0.6578336575729304
Lars 의 acc :  0.7156778188107706
LarsCV 의 acc :  0.7255385058810935
Lasso 의 acc :  0.342471641466883
LassoCV 의 acc :  0.7215483085857873
LassoLars 의 acc :  -0.014319569088965922
LassoLarsCV 의 acc :  0.7196372557659261
LassoLarsIC 의 acc :  0.7182403396119904
LinearRegression 의 acc :  0.7159952162619656
LinearSVR 의 acc :  0.6425338317571643
MLPRegressor 의 acc :  0.12064532578730014
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 의 acc :  0.6118115608281913
OrthogonalMatchingPursuit 의 acc :  0.4765510501802527
OrthogonalMatchingPursuitCV 의 acc :  0.6891771491719336
PLSCanonical 의 acc :  -3.211630124846253
PLSRegression 의 acc :  0.7490133018126226
PassiveAggressiveRegressor 의 acc :  0.7585196787410856
PoissonRegressor 의 acc :  0.650013404250769
RANSACRegressor 의 acc :  0.6427590167639392
RadiusNeighborsRegressor 의 acc :  0.3321024717483456
RandomForestRegressor 의 acc :  0.8636771069340384
RegressorChain 없는것
Ridge 의 acc :  0.7301885676967046
RidgeCV 의 acc :  0.7187847665990533
SGDRegressor 의 acc :  0.7210791574553757
SVR 의 acc :  0.6455875874652905
StackingRegressor 없는것
TheilSenRegressor 의 acc :  0.7571715362780473
TransformedTargetRegressor 의 acc :  0.7159952162619656
TweedieRegressor 의 acc :  0.19021067033855799
VotingRegressor 없는것
===============================================================

StandardScaler
ARDRegression 의 acc :  0.7207168832130914
AdaBoostRegressor 의 acc :  0.8203510376906169
BaggingRegressor 의 acc :  0.8539696186533101
BayesianRidge 의 acc :  0.7220699524590716
CCA 의 acc :  0.623687582029342
DecisionTreeRegressor 의 acc :  0.7597704474421448
DummyRegressor 의 acc :  -0.014319569088965922
ElasticNet 의 acc :  0.6890709912539126
ElasticNetCV 의 acc :  0.7260284022716572
ExtraTreeRegressor 의 acc :  0.7420142059658883
ExtraTreesRegressor 의 acc :  0.8830533668938467
GammaRegressor 의 acc :  0.6851519031174427
GaussianProcessRegressor 의 acc :  0.4353368832508927
GradientBoostingRegressor 의 acc :  0.8763374219807245
HistGradientBoostingRegressor 의 acc :  0.8608461771215346
HuberRegressor 의 acc :  0.7464389240830045
IsotonicRegression 없는것
KNeighborsRegressor 의 acc :  0.7287828630981212
KernelRidge 의 acc :  -7.045992533531814
Lars 의 acc :  0.7156778188107706
LarsCV 의 acc :  0.7255385058810935
Lasso 의 acc :  0.6903178291309635
LassoCV 의 acc :  0.7197334859523027
LassoLars 의 acc :  -0.014319569088965922
LassoLarsCV 의 acc :  0.7196372557659254
LassoLarsIC 의 acc :  0.7182403396119904
LinearRegression 의 acc :  0.7159952162619658
LinearSVR 의 acc :  0.7394999102503752
MLPRegressor 의 acc :  0.6697186552783139
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 의 acc :  0.6767520605657991
OrthogonalMatchingPursuit 의 acc :  0.4765510501802527
OrthogonalMatchingPursuitCV 의 acc :  0.6891771491719336
PLSCanonical 의 acc :  -3.2116301248462573
PLSRegression 의 acc :  0.7490133018126225
PassiveAggressiveRegressor 의 acc :  0.35483070861247623
PoissonRegressor 의 acc :  0.812103359049245
RANSACRegressor 의 acc :  0.16210290688068352
RadiusNeighborsRegressor 없는것
RandomForestRegressor 의 acc :  0.8618422779047761
RegressorChain 없는것
Ridge 의 acc :  0.7173525038637134
RidgeCV 의 acc :  0.7173525038638984
SGDRegressor 의 acc :  0.7159628404584747
SVR 의 acc :  0.6854386986018963
StackingRegressor 없는것
TheilSenRegressor 의 acc :  0.6641330250071776
TransformedTargetRegressor 의 acc :  0.7159952162619658
TweedieRegressor 의 acc :  0.6709589351580008
VotingRegressor 없는것
'''