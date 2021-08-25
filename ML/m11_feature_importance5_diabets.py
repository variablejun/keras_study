# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

dataset = load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,
train_size = 0.8, random_state=76)

model = DecisionTreeRegressor(max_depth=4)
#model = RandomForestRegressor()

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
DecisionTreeRegressor
acc : -0.27665037727342945
[0.0961046  0.01876816 0.38060894 0.04392787 0.04916982 0.06268684
 0.05614813 0.00197087 0.22226894 0.06834584]

max_depth=4
acc : 0.248529478678701
[0.02103787 0.01580793 0.56205071 0.01888243 0.         0.03151006
 0.03762198 0.         0.27497255 0.03811648]

RandomForestRegressor
acc : 0.3406874018020327
[0.06159503 0.01073445 0.32070481 0.1036172  0.04209233 0.05330236
 0.04859934 0.02759982 0.26427335 0.0674813 ]
'''
