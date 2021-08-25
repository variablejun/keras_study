# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(dataset.data,dataset.target,
train_size = 0.7, random_state=76)

model = XGBClassifier()
#model = RandomForestClassifier()

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
acc : 0.9122807017543859
[0.00941421 0.00356721 0.         0.02712885 0.         0.
 0.         0.         0.         0.         0.         0.01965424
 0.         0.01034948 0.         0.         0.         0.
 0.         0.         0.02258095 0.02185279 0.129193   0.00855838
 0.0105249  0.         0.01070481 0.70764276 0.         0.01882843]
 
max_depth=4
acc : 0.9239766081871345
[0.         0.04068436 0.         0.         0.         0.
 0.         0.         0.         0.         0.0111966  0.01682086
 0.         0.01131726 0.         0.         0.0109416  0.
 0.         0.         0.01319879 0.         0.13658449 0.
 0.         0.         0.01112706 0.74812898 0.         0.        ]

RandomForestClassifier
acc : 0.9649122807017544
[0.0365405  0.019424   0.02831467 0.02732332 0.00512679 0.00648297
 0.05963969 0.09164334 0.00445291 0.00615097 0.00822849 0.00410203
 0.01699811 0.03628441 0.00510202 0.00342139 0.0037871  0.0039241
 0.00339757 0.00357266 0.08391388 0.0174914  0.13528848 0.134535
 0.01154565 0.02450616 0.03576239 0.16086695 0.0148681  0.00730493]

GradientBoostingClassifier
 acc : 0.9532163742690059
[1.21699013e-03 3.32240451e-02 3.56386656e-04 3.22027761e-03
 1.02975601e-03 3.07215238e-03 3.64634180e-03 3.05908805e-02
 6.72315396e-04 2.31107286e-04 5.10532887e-04 6.50060568e-03
 1.64516243e-03 8.48616645e-03 5.67576984e-05 5.20922244e-03
 4.58325194e-05 3.81839730e-04 3.11007581e-03 6.45700070e-03
 2.41362657e-01 1.94668663e-02 1.82908953e-01 2.95637119e-02
 3.50540852e-03 2.77101937e-04 8.43056701e-03 4.02101816e-01
 1.50088788e-03 1.21858066e-03]

acc : 0.9649122807017544
[0.         0.01811066 0.         0.01029031 0.00176827 0.00425797
 0.00269142 0.06227905 0.         0.02041085 0.         0.
 0.         0.01100578 0.00459318 0.00285144 0.         0.00391256
 0.00124341 0.00175978 0.03563771 0.01734336 0.09207205 0.22841732
 0.0075046  0.05010894 0.01670291 0.39265087 0.00652667 0.00786091]
'''
