# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

dataset = load_wine()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,
train_size = 0.7, random_state=76)

#model = DecisionTreeClassifier(max_depth=4)
model = RandomForestClassifier()

model.fit(x_train, y_train)

acc = model.score(x_test,y_test)
print('acc :', acc)
print(model.feature_importances_) # iris의 컬럼중에서 어느 컬럼이 영향이큰지 영향도를 나타내는것
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
DecisionTreeClassifier
acc : 0.9814814814814815
[0.03838585 0.         0.         0.         0.02411578 0.0901017
 0.1048353  0.         0.         0.29042927 0.01974129 0.
 0.43239081]
 
max_depth=4
acc : 0.9259259259259259
[0.         0.0390279  0.         0.02451915 0.         0.09160876
 0.1065888  0.         0.         0.31535855 0.         0.
 0.42289684]

RandomForestClassifier
acc : 0.9814814814814815
[0.13992509 0.04635975 0.01922157 0.01933676 0.0203586  0.06301032
 0.16708597 0.00884209 0.02110244 0.15958649 0.10162195 0.08347303
 0.15007595]
'''
