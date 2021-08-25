#피처임포턴스에서 나온 값중 중요도 20% 이하는 컬럼제거후 데이터셋재구성후 다시돌려서 모델구성
# 3가지 부스트모델 적용
# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
print(x.shape) # (569, 30)
'''
print(dataset.feature_names)

[ 'mean texture'  'mean area'
  'mean concavity'
 'mean concave points' 'mean symmetry' 'mean fractal dimension'
 'radius error' 'texture error' 'perimeter error' 'area error'
 'smoothness error' 'compactness error' 'concavity error'
 'concave points error' 'symmetry error' 'fractal dimension error'
 'worst radius' 'worst texture' 'worst perimeter' 'worst area'
 'worst smoothness' 'worst compactness' 'worst concavity'
 'worst concave points' ]
'''
dataset = pd.DataFrame(dataset.data,columns=dataset.feature_names)

x = dataset[[ 'mean texture' ,'mean area'
  ,'mean concavity',
 'mean concave points' ,'mean symmetry' ,'mean fractal dimension',
 'radius error' ,'texture error' ,'perimeter error' ,'area error',
 'smoothness error' ,'compactness error', 'concavity error',
 'concave points error', 'symmetry error', 'fractal dimension error',
 'worst radius' ,'worst texture', 'worst perimeter', 'worst area',
 'worst smoothness' ,'worst compactness', 'worst concavity',
 'worst concave points' ]]


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

model = RandomForestClassifier()
#model = XGBClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print('acc :', acc)
print(model.feature_importances_) # iris의 컬럼중에서 어느 컬럼이 영향이큰지 영향도를 나타내는것
'''
import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
     n_feature = dataset.data.shape[1]
     plt.barh(np.arange(n_feature),model.feature_importances_,align='center')
     plt.yticks(np.arange(n_feature),dataset.feature_names)
     plt.xlabel('feature_importances')
     plt.ylabel('features')
     plt.ylim(-1,n_feature)
plot_feature_importances_dataset(model)   
plt.show()     
'''

'''
XGBClassifier
acc : 0.9707602339181286
[0.01802825 0.01854027 0.00053494 0.06723569 0.         0.01235769
 0.00186569 0.00982154 0.00146577 0.01006427 0.00537209 0.0023558
 0.         0.00623052 0.         0.00336102 0.02809021 0.01958269
 0.1210876  0.34216663 0.00719956 0.05485289 0.01798658 0.25180033]

DecisionTreeClassifier
acc : 0.935672514619883
[0.02322144 0.         0.         0.01856233 0.         0.
 0.         0.02598727 0.00125523 0.02011902 0.         0.00732217
 0.         0.         0.01059068 0.01034948 0.01248452 0.0227472
 0.129193   0.         0.         0.         0.0105249  0.70764276]

GradientBoostingClassifier
acc : 0.9649122807017544
[3.64579098e-02 4.42666172e-03 4.83118948e-03 3.24739834e-02
 6.24468248e-04 4.54827930e-04 7.78956982e-04 6.69629802e-03
 2.15763909e-03 8.01874060e-03 7.51732467e-06 4.94487779e-03
 4.48547033e-03 2.30788056e-03 5.33700404e-03 2.43293941e-03
 2.40797429e-01 1.83430711e-02 1.82668783e-01 3.11339164e-02
 3.63085971e-03 7.25291077e-05 7.09403244e-03 3.99823015e-01]

RandomForestClassifier
acc : 0.9766081871345029
[0.02405992 0.07194974 0.05771771 0.10196869 0.00504383 0.00541754
 0.03342522 0.00456438 0.02215315 0.05716476 0.00536861 0.00549288
 0.00624238 0.00517394 0.00619356 0.00660144 0.1044553  0.02546158
 0.13871078 0.09784657 0.0094631  0.03218422 0.05967356 0.11366712]
 
'''
