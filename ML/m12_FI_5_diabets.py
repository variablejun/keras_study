#피처임포턴스에서 나온 값중 중요도 20% 이하는 컬럼제거후 데이터셋재구성후 다시돌려서 모델구성
# 3가지 부스트모델 적용
# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_diabetes()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
print(x.shape) # (569, 30)



'''

[ 'sex', 'bmi', 'bp', 's2', 's3', 's5', 's6']
'''
dataset = pd.DataFrame(dataset.data,columns=dataset.feature_names)

x = dataset[[  'sex', 'bmi', 'bp', 's2', 's3', 's5', 's6']]


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

model = XGBRegressor()
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
XGBRegressor
acc : 0.23793876014528448
[0.03728111 0.1905041  0.11703609 0.06710985 0.09546072 0.4050486
 0.08755957]

DecisionTreeRegressor
acc : -0.27321040799310947
[0.00678099 0.25103057 0.12726668 0.12934949 0.0515264  0.34115476
 0.09289112]

GradientBoostingRegressor
acc : 0.4185927605214119
[0.01754524 0.30537143 0.13340088 0.07239832 0.0592768  0.33671012
 0.0752972 ]

RandomForestRegressor
acc : 0.42588689505427135
[0.01342232 0.29710699 0.1284542  0.08078989 0.07645267 0.31197563
 0.0917983 ]
'''
