#피처임포턴스에서 나온 값중 중요도 20% 이하는 컬럼제거후 데이터셋재구성후 다시돌려서 모델구성
# 3가지 부스트모델 적용
# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_iris()

y = dataset.target

print(dataset.feature_names)
#['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
dataset = pd.DataFrame(dataset.data,columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

x = dataset[['sepal length (cm)','sepal width (cm)','petal width (cm)']]

x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

model = RandomForestClassifier()
#model = XGBClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print('acc :', acc)
print(model.feature_importances_) # iris의 컬럼중에서 어느 컬럼이 영향이큰지 영향도를 나타내는것

import numpy as np
import matplotlib.pyplot as plt
'''
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
acc : 0.9777777777777777
[0.02828696 0.03890036 0.9328127 ]

DecisionTreeClassifier
acc : 0.9555555555555556
[0.02562906 0.00423429 0.97013665]

GradientBoostingClassifier
acc : 0.9555555555555556
[0.04697177 0.02675355 0.92627468]

RandomForestClassifier
acc : 0.9555555555555556
[0.2840261  0.18520968 0.53076422]
'''
