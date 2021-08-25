import numpy as np 
from sklearn.datasets import load_iris

dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target

print(x.shape , y.shape)
'''
from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
머신러닝에선 원핫인코딩 적용안해도 해줌
ValueError: y should be a 1d array, got an array of shape (142, 3) instead.
1차원을 줘야하는데 2차원을줫음 원핫인코딩 때문
대부분의 머신러닝들을 y를 1차원으로 받아들여서 오류가나고 따로 안해줘도됨
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.95, random_state=76)

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
          acc = accuracy_score(y_test,y_predict)
          print(name,'의 acc : ',acc) # tap 한칸 shift + tap 한칸뒤로
     except:
          print(name,'없는것')
#여러가지 경우에 따라서 파라미터가 안맞거나 해서 fit을 못할경우 따로 빼서 이름만 출력

     



'''
     [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ('BaggingClassifier', <class 'sklearn.ensemble._bagging.BaggingClassifier'>), ('BernoulliNB', <class 'sklearn.naive_bayes.BernoulliNB'>), ('CalibratedClassifierCV', <class 'sklearn.calibration.CalibratedClassifierCV'>), ('CategoricalNB', <class 'sklearn.naive_bayes.CategoricalNB'>), ('ClassifierChain', <class 'sklearn.multioutput.ClassifierChain'>), ('ComplementNB', <class 'sklearn.naive_bayes.ComplementNB'>), ('DecisionTreeClassifier', <class 'sklearn.tree._classes.DecisionTreeClassifier'>), ('DummyClassifier', <class 'sklearn.dummy.DummyClassifier'>), ('ExtraTreeClassifier', <class 'sklearn.tree._classes.ExtraTreeClassifier'>), ('ExtraTreesClassifier', <class 'sklearn.ensemble._forest.ExtraTreesClassifier'>), ('GaussianNB', <class 'sklearn.naive_bayes.GaussianNB'>), ('GaussianProcessClassifier', <class 'sklearn.gaussian_process._gpc.GaussianProcessClassifier'>), ('GradientBoostingClassifier', <class 'sklearn.ensemble._gb.GradientBoostingClassifier'>), ('HistGradientBoostingClassifier', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>), ('KNeighborsClassifier', <class 'sklearn.neighbors._classification.KNeighborsClassifier'>), ('LabelPropagation', <class 'sklearn.semi_supervised._label_propagation.LabelPropagation'>), ('LabelSpreading', <class 'sklearn.semi_supervised._label_propagation.LabelSpreading'>), ('LinearDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.LinearDiscriminantAnalysis'>), ('LinearSVC', <class 'sklearn.svm._classes.LinearSVC'>), ('LogisticRegression', 
     <class 'sklearn.linear_model._logistic.LogisticRegression'>), ('LogisticRegressionCV', <class 'sklearn.linear_model._logistic.LogisticRegressionCV'>), ('MLPClassifier', <class 'sklearn.neural_network._multilayer_perceptron.MLPClassifier'>), ('MultiOutputClassifier', <class 'sklearn.multioutput.MultiOutputClassifier'>), ('MultinomialNB', <class 'sklearn.naive_bayes.MultinomialNB'>), ('NearestCentroid', <class 
     'sklearn.neighbors._nearest_centroid.NearestCentroid'>), ('NuSVC', <class 'sklearn.svm._classes.NuSVC'>), ('OneVsOneClassifier', <class 'sklearn.multiclass.OneVsOneClassifier'>), ('OneVsRestClassifier', <class 'sklearn.multiclass.OneVsRestClassifier'>), ('OutputCodeClassifier', <class 'sklearn.multiclass.OutputCodeClassifier'>), ('PassiveAggressiveClassifier', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveClassifier'>), ('Perceptron', <class 'sklearn.linear_model._perceptron.Perceptron'>), ('QuadraticDiscriminantAnalysis', <class 'sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis'>), ('RadiusNeighborsClassifier', <class 'sklearn.neighbors._classification.RadiusNeighborsClassifier'>), ('RandomForestClassifier', <class 'sklearn.ensemble._forest.RandomForestClassifier'>), ('RidgeClassifier', <class 'sklearn.linear_model._ridge.RidgeClassifier'>), ('RidgeClassifierCV', <class 'sklearn.linear_model._ridge.RidgeClassifierCV'>), ('SGDClassifier', <class 'sklearn.linear_model._stochastic_gradient.SGDClassifier'>), ('SVC', <class 'sklearn.svm._classes.SVC'>), ('StackingClassifier', <class 'sklearn.ensemble._stacking.StackingClassifier'>), ('VotingClassifier', <class 'sklearn.ensemble._voting.VotingClassifier'>)]
     [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ('AdaBoostRegressor', <class 'sklearn.ensemble._weight_boosting.AdaBoostRegressor'>), ('BaggingRegressor', <class 'sklearn.ensemble._bagging.BaggingRegressor'>), ('BayesianRidge', <class 'sklearn.linear_model._bayes.BayesianRidge'>), ('CCA', <class 'sklearn.cross_decomposition._pls.CCA'>), ('DecisionTreeRegressor', <class 'sklearn.tree._classes.DecisionTreeRegressor'>), ('DummyRegressor', <class 'sklearn.dummy.DummyRegressor'>), ('ElasticNet', <class 'sklearn.linear_model._coordinate_descent.ElasticNet'>), ('ElasticNetCV', <class 'sklearn.linear_model._coordinate_descent.ElasticNetCV'>), ('ExtraTreeRegressor', <class 'sklearn.tree._classes.ExtraTreeRegressor'>), ('ExtraTreesRegressor', <class 'sklearn.ensemble._forest.ExtraTreesRegressor'>), ('GammaRegressor', <class 'sklearn.linear_model._glm.glm.GammaRegressor'>), ('GaussianProcessRegressor', <class 'sklearn.gaussian_process._gpr.GaussianProcessRegressor'>), ('GradientBoostingRegressor', <class 'sklearn.ensemble._gb.GradientBoostingRegressor'>), ('HistGradientBoostingRegressor', <class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor'>), ('HuberRegressor', <class 'sklearn.linear_model._huber.HuberRegressor'>), ('IsotonicRegression', <class 'sklearn.isotonic.IsotonicRegression'>), ('KNeighborsRegressor', <class 'sklearn.neighbors._regression.KNeighborsRegressor'>), ('KernelRidge', <class 'sklearn.kernel_ridge.KernelRidge'>), ('Lars', <class 'sklearn.linear_model._least_angle.Lars'>), ('LarsCV', <class 'sklearn.linear_model._least_angle.LarsCV'>), ('Lasso', <class 'sklearn.linear_model._coordinate_descent.Lasso'>), ('LassoCV', <class 'sklearn.linear_model._coordinate_descent.LassoCV'>), ('LassoLars', <class 'sklearn.linear_model._least_angle.LassoLars'>), ('LassoLarsCV', <class 'sklearn.linear_model._least_angle.LassoLarsCV'>), ('LassoLarsIC', <class 'sklearn.linear_model._least_angle.LassoLarsIC'>), ('LinearRegression', <class 'sklearn.linear_model._base.LinearRegression'>), ('LinearSVR', <class 'sklearn.svm._classes.LinearSVR'>), ('MLPRegressor', <class 'sklearn.neural_network._multilayer_perceptron.MLPRegressor'>), ('MultiOutputRegressor', <class 'sklearn.multioutput.MultiOutputRegressor'>), ('MultiTaskElasticNet', <class 'sklearn.linear_model._coordinate_descent.MultiTaskElasticNet'>), ('MultiTaskElasticNetCV', <class 'sklearn.linear_model._coordinate_descent.MultiTaskElasticNetCV'>), ('MultiTaskLasso', <class 'sklearn.linear_model._coordinate_descent.MultiTaskLasso'>), ('MultiTaskLassoCV', <class 'sklearn.linear_model._coordinate_descent.MultiTaskLassoCV'>), ('NuSVR', <class 'sklearn.svm._classes.NuSVR'>), ('OrthogonalMatchingPursuit', <class 'sklearn.linear_model._omp.OrthogonalMatchingPursuit'>), ('OrthogonalMatchingPursuitCV', <class 'sklearn.linear_model._omp.OrthogonalMatchingPursuitCV'>), ('PLSCanonical', <class 'sklearn.cross_decomposition._pls.PLSCanonical'>), ('PLSRegression', <class 'sklearn.cross_decomposition._pls.PLSRegression'>), ('PassiveAggressiveRegressor', <class 'sklearn.linear_model._passive_aggressive.PassiveAggressiveRegressor'>), ('PoissonRegressor', <class 'sklearn.linear_model._glm.glm.PoissonRegressor'>), ('RANSACRegressor', <class 'sklearn.linear_model._ransac.RANSACRegressor'>), ('RadiusNeighborsRegressor', <class 'sklearn.neighbors._regression.RadiusNeighborsRegressor'>), ('RandomForestRegressor', <class 'sklearn.ensemble._forest.RandomForestRegressor'>), ('RegressorChain', <class 'sklearn.multioutput.RegressorChain'>), ('Ridge', <class 'sklearn.linear_model._ridge.Ridge'>), ('RidgeCV', <class 'sklearn.linear_model._ridge.RidgeCV'>), ('SGDRegressor', <class 'sklearn.linear_model._stochastic_gradient.SGDRegressor'>), ('SVR', <class 'sklearn.svm._classes.SVR'>), ('StackingRegressor', <class 'sklearn.ensemble._stacking.StackingRegressor'>), ('TheilSenRegressor', <class 'sklearn.linear_model._theil_sen.TheilSenRegressor'>), ('TransformedTargetRegressor', <class 'sklearn.compose._target.TransformedTargetRegressor'>), ('TweedieRegressor', <class 'sklearn.linear_model._glm.glm.TweedieRegressor'>), ('VotingRegressor', <class 'sklearn.ensemble._voting.VotingRegressor'>)]
'''
'''
회귀모델
MinMaxScaler
ARDRegression 없는것
AdaBoostRegressor 없는것
BaggingRegressor 없는것
BayesianRidge 없는것
CCA 없는것
DecisionTreeRegressor 의 acc :  1.0
DummyRegressor 없는것
ElasticNet 없는것
ElasticNetCV 없는것
ExtraTreeRegressor 의 acc :  1.0
ExtraTreesRegressor 없는것
GammaRegressor 없는것
GaussianProcessRegressor 없는것
GradientBoostingRegressor 없는것
HistGradientBoostingRegressor 없는것
HuberRegressor 없는것
IsotonicRegression 없는것
KNeighborsRegressor 없는것
KernelRidge 없는것
Lars 없는것
LarsCV 없는것
Lasso 없는것
LassoCV 없는것
LassoLars 없는것
LassoLarsCV 없는것
LassoLarsIC 없는것
LinearRegression 없는것
LinearSVR 없는것
MLPRegressor 없는것
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 없는것
OrthogonalMatchingPursuit 없는것
OrthogonalMatchingPursuitCV 없는것
PLSCanonical 없는것
PLSRegression 없는것
PassiveAggressiveRegressor 없는것
PoissonRegressor 없는것
RANSACRegressor 없는것
RadiusNeighborsRegressor 없는것
RandomForestRegressor 없는것
RegressorChain 없는것
Ridge 없는것
RidgeCV 없는것
SGDRegressor 없는것
SVR 없는것
StackingRegressor 없는것
TheilSenRegressor 없는것
TransformedTargetRegressor 없는것
TweedieRegressor 없는것
VotingRegressor 없는것

StandardScaler
ARDRegression 없는것
AdaBoostRegressor 없는것
BaggingRegressor 없는것
BayesianRidge 없는것
CCA 없는것
DecisionTreeRegressor 의 acc :  1.0
DummyRegressor 없는것
ElasticNet 없는것
ElasticNetCV 없는것
ExtraTreeRegressor 의 acc :  1.0
ExtraTreesRegressor 없는것
GammaRegressor 없는것
GaussianProcessRegressor 없는것
GradientBoostingRegressor 없는것
HistGradientBoostingRegressor 없는것
HuberRegressor 없는것
IsotonicRegression 없는것
KNeighborsRegressor 없는것
KernelRidge 없는것
Lars 없는것
LarsCV 없는것
Lasso 없는것
LassoCV 없는것
LassoLars 없는것
LassoLarsCV 없는것
LassoLarsIC 없는것
LinearRegression 없는것
LinearSVR 없는것
MLPRegressor 없는것
MultiOutputRegressor 없는것
MultiTaskElasticNet 없는것
MultiTaskElasticNetCV 없는것
MultiTaskLasso 없는것
MultiTaskLassoCV 없는것
NuSVR 없는것
OrthogonalMatchingPursuit 없는것
OrthogonalMatchingPursuitCV 없는것
PLSCanonical 없는것
PLSRegression 없는것
PassiveAggressiveRegressor 없는것
PoissonRegressor 없는것
RANSACRegressor 없는것
RadiusNeighborsRegressor 없는것
RandomForestRegressor 없는것
RegressorChain 없는것
Ridge 없는것
RidgeCV 없는것
SGDRegressor 없는것
SVR 없는것
StackingRegressor 없는것
TheilSenRegressor 없는것
TransformedTargetRegressor 없는것
TweedieRegressor 없는것
VotingRegressor 없는것

분류모델
MinMaxScaler
AdaBoostClassifier 의 acc :  1.0
BaggingClassifier 의 acc :  0.875
BernoulliNB 의 acc :  0.5
CalibratedClassifierCV 의 acc :  1.0
CategoricalNB 의 acc :  0.375
ClassifierChain 없는것
ComplementNB 의 acc :  0.625
DecisionTreeClassifier 의 acc :  1.0
DummyClassifier 의 acc :  0.25
ExtraTreeClassifier 의 acc :  0.875
ExtraTreesClassifier 의 acc :  0.875
GaussianNB 의 acc :  1.0
GaussianProcessClassifier 의 acc :  1.0
GradientBoostingClassifier 의 acc :  1.0
HistGradientBoostingClassifier 의 acc :  0.875
KNeighborsClassifier 의 acc :  1.0
LabelPropagation 의 acc :  1.0
LabelSpreading 의 acc :  1.0
LinearDiscriminantAnalysis 의 acc :  1.0
LinearSVC 의 acc :  1.0
LogisticRegression 의 acc :  1.0
LogisticRegressionCV 의 acc :  1.0
MLPClassifier 의 acc :  0.875
MultiOutputClassifier 없는것
MultinomialNB 의 acc :  0.625
NearestCentroid 의 acc :  1.0
NuSVC 의 acc :  1.0
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  0.625
Perceptron 의 acc :  0.625
QuadraticDiscriminantAnalysis 의 acc :  1.0
RadiusNeighborsClassifier 의 acc :  0.625
RandomForestClassifier 의 acc :  1.0
RidgeClassifier 의 acc :  1.0
RidgeClassifierCV 의 acc :  1.0
SGDClassifier 의 acc :  1.0
SVC 의 acc :  1.0
StackingClassifier 없는것
VotingClassifier 없는것

=========================================================
StandardScaler
AdaBoostClassifier 의 acc :  1.0
BaggingClassifier 의 acc :  1.0
BernoulliNB 의 acc :  0.75
CalibratedClassifierCV 의 acc :  1.0
CategoricalNB 없는것
ClassifierChain 없는것
ComplementNB 없는것
DecisionTreeClassifier 의 acc :  1.0
DummyClassifier 의 acc :  0.25
ExtraTreeClassifier 의 acc :  1.0
ExtraTreesClassifier 의 acc :  0.875
GaussianNB 의 acc :  1.0
GaussianProcessClassifier 의 acc :  1.0
GradientBoostingClassifier 의 acc :  1.0
HistGradientBoostingClassifier 의 acc :  0.875
KNeighborsClassifier 의 acc :  0.875
LabelPropagation 의 acc :  0.875
LabelSpreading 의 acc :  0.875
LinearDiscriminantAnalysis 의 acc :  1.0
LinearSVC 의 acc :  1.0
LogisticRegression 의 acc :  1.0
LogisticRegressionCV 의 acc :  1.0
MLPClassifier 의 acc :  1.0
MultiOutputClassifier 없는것
MultinomialNB 없는것
NearestCentroid 의 acc :  1.0
NuSVC 의 acc :  1.0
OneVsOneClassifier 없는것
OneVsRestClassifier 없는것
OutputCodeClassifier 없는것
PassiveAggressiveClassifier 의 acc :  1.0
Perceptron 의 acc :  1.0
QuadraticDiscriminantAnalysis 의 acc :  1.0
RadiusNeighborsClassifier 의 acc :  0.875
RandomForestClassifier 의 acc :  0.875
RidgeClassifier 의 acc :  1.0
RidgeClassifierCV 의 acc :  1.0
SGDClassifier 의 acc :  0.875
SVC 의 acc :  1.0
StackingClassifier 없는것
VotingClassifier 없는것

스케일러에 따라서 달라질 수 있다

'''