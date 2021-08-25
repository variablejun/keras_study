import numpy as np 
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
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
ARDRegression 의 acc :  0.4752240742655116
AdaBoostRegressor 의 acc :  0.4279595068668437
BaggingRegressor 의 acc :  0.3649889071057143
BayesianRidge 의 acc :  0.4779003988432058
CCA 의 acc :  0.45663223888132476
DecisionTreeRegressor 의 acc :  -0.08852746591384464
DummyRegressor 의 acc :  -0.01828713109513047
ElasticNet 의 acc :  0.1160314289225125
ElasticNetCV 의 acc :  0.4667478524328301
ExtraTreeRegressor 의 acc :  -0.6177008764659344
ExtraTreesRegressor 의 acc :  0.3532142774758431
GammaRegressor 의 acc :  0.05751789726483236
GaussianProcessRegressor 의 acc :  -11.146362256316994
GradientBoostingRegressor 의 acc :  0.37845025663469856
HistGradientBoostingRegressor 의 acc :  0.3622106724151084
HuberRegressor 의 acc :  0.48211251161269153
IsotonicRegression 없는것
KNeighborsRegressor 의 acc :  0.22170329564005076
KernelRidge 의 acc :  0.4655777781536714
Lars 의 acc :  0.4836152351755426
LarsCV 의 acc :  0.4717723038473799
Lasso 의 acc :  0.4643959880288301
LassoCV 의 acc :  0.47235279285723875
LassoLars 의 acc :  0.3789574704455142
LassoLarsCV 의 acc :  0.47256824911735984
LassoLarsIC 의 acc :  0.46899632465550034
LinearRegression 의 acc :  0.4836152351755424
LinearSVR 의 acc :  0.033389405467916244
MLPRegressor 의 acc :  -0.8018296939480816
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 의 acc :  0.0945548726580927
OrthogonalMatchingPursuit 의 acc :  0.331571344245463
OrthogonalMatchingPursuitCV 의 acc :  0.47236888137237576
PLSCanonical 의 acc :  -1.0813289977302305
PLSRegression 의 acc :  0.4717068711385083
PassiveAggressiveRegressor 의 acc :  0.46768266421594984
PoissonRegressor 의 acc :  0.4546079389735007
RANSACRegressor 의 acc :  0.10691621430925724
RadiusNeighborsRegressor 의 acc :  0.13626339730156645
RandomForestRegressor 의 acc :  0.3856562236051718
RegressorChain 없는것
Ridge 의 acc :  0.47645839809155244
RidgeCV 의 acc :  0.47645839809155266
SGDRegressor 의 acc :  0.47286228706636246
SVR 의 acc :  0.0722496823999842
StackingRegressor 없는것
TheilSenRegressor 의 acc :  0.4791477446216086
TransformedTargetRegressor 의 acc :  0.4836152351755424
TweedieRegressor 의 acc :  0.06516671638383154
VotingRegressor 없는것
===============================================================

StandardScaler
ARDRegression 의 acc :  0.47522757597157683
AdaBoostRegressor 의 acc :  0.4015756084178954
BaggingRegressor 의 acc :  0.3513341052382415
BayesianRidge 의 acc :  0.4774020721525273
CCA 의 acc :  0.45663223888132465
DecisionTreeRegressor 의 acc :  -0.08823587712256642
DummyRegressor 의 acc :  -0.01828713109513047
ElasticNet 의 acc :  0.43524089513852116
ElasticNetCV 의 acc :  0.4745849650452888
ExtraTreeRegressor 의 acc :  -0.26648699702065937
ExtraTreesRegressor 의 acc :  0.36663774231512336
GammaRegressor 의 acc :  0.3775423132113169
GaussianProcessRegressor 의 acc :  -0.9437763126457914
GradientBoostingRegressor 의 acc :  0.3780709983381272
HistGradientBoostingRegressor 의 acc :  0.36225196424060746
HuberRegressor 의 acc :  0.48219265242926357
IsotonicRegression 없는것
KNeighborsRegressor 의 acc :  0.24877078428524657
KernelRidge 의 acc :  -3.5768889671767674
Lars 의 acc :  0.4836152351755425
LarsCV 의 acc :  0.4717723038473802
Lasso 의 acc :  0.4780798348004758
LassoCV 의 acc :  0.47025743495871386
LassoLars 의 acc :  0.3789574704455144
LassoLarsCV 의 acc :  0.47256824911735984
LassoLarsIC 의 acc :  0.46899632465550034
LinearRegression 의 acc :  0.4836152351755426
LinearSVR 의 acc :  0.18908039524403253
MLPRegressor 의 acc :  -1.3615960918631203
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 의 acc :  0.1137402566699709
OrthogonalMatchingPursuit 의 acc :  0.3315713442454632
OrthogonalMatchingPursuitCV 의 acc :  0.472368881372376
PLSCanonical 의 acc :  -1.0813289977302283
PLSRegression 의 acc :  0.4717068711385085
PassiveAggressiveRegressor 의 acc :  0.4612170843801403
PoissonRegressor 의 acc :  0.4700833580675722
RANSACRegressor 의 acc :  -1.3359040401217293
RadiusNeighborsRegressor 없는것
RandomForestRegressor 의 acc :  0.3868673521086945
RegressorChain 없는것
Ridge 의 acc :  0.48275105639679483
RidgeCV 의 acc :  0.478819640170429
SGDRegressor 의 acc :  0.47796331732119757
SVR 의 acc :  0.09003478179417812
StackingRegressor 없는것
TheilSenRegressor 의 acc :  0.47651355010260654
TransformedTargetRegressor 의 acc :  0.4836152351755426
TweedieRegressor 의 acc :  0.39844973465227707
VotingRegressor 없는것
'''