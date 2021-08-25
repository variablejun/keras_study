#피처임포턴스에서 나온 값중 중요도 20% 이하는 컬럼제거후 데이터셋재구성후 다시돌려서 모델구성
# 3가지 부스트모델 적용
# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_wine()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
print(x.shape) # (569, 30)
print(dataset.feature_names)


'''

['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
  'proanthocyanins', 'color_intensity', 'proline']
'''
dataset = pd.DataFrame(dataset.data,columns=dataset.feature_names)

x = dataset[[ 'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols',
  'proanthocyanins', 'color_intensity', 'proline']]


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

model = XGBClassifier()
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
acc : 0.9629629629629629
[0.04532846 0.09011773 0.05196821 0.05039956 0.02213488 0.07857756
 0.17193744 0.24628432 0.2432518 ]

DecisionTreeClassifier
acc : 0.8518518518518519
[0.         0.07124768 0.01233831 0.01974129 0.02411578 0.12711662
 0.         0.31099311 0.43444719]

GradientBoostingClassifier
acc : 0.9629629629629629
[0.01511766 0.05904526 0.02461629 0.05404478 0.00353364 0.06088015
 0.07581182 0.40533174 0.30161867]

RandomForestClassifier
acc : 0.9444444444444444
[0.19067115 0.06695329 0.02725023 0.03287883 0.03679971 0.12962535
 0.04417902 0.22198289 0.24965953]

'''
