# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

dataset = load_boston()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,
train_size = 0.8, random_state=76)

#model = DecisionTreeClassifier(max_depth=4)
model = RandomForestRegressor()

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
acc : 0.8888888888888888
[0.01433251 0.03439803 0.909934   0.04133545]

max_depth=4
acc : 0.7054300005529629
[2.58965211e-02 4.91603046e-05 2.27330004e-03 2.26572547e-05
 6.04700340e-03 2.48703427e-01 1.32248851e-02 1.00543382e-01
 4.28488496e-03 1.74469531e-02 1.55813574e-02 1.15196239e-02
 5.54406845e-01]

RandomForestRegressor
acc : 0.8632106564568358
[0.03338373 0.00154591 0.00804222 0.00116112 0.01313813 0.35542673
 0.0152941  0.07121269 0.00470502 0.01568824 0.01536014 0.01124069
 0.45380128]
'''
