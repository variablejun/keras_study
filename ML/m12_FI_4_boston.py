#피처임포턴스에서 나온 값중 중요도 20% 이하는 컬럼제거후 데이터셋재구성후 다시돌려서 모델구성
# 3가지 부스트모델 적용
# 피처 = 열 = 컬럼 = 특성

from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRFRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd


dataset = load_boston()
x = dataset.data
y = dataset.target
print(dataset.feature_names)
print(x.shape) # (569, 30)



'''

['CRIM' ,'ZN' ,'INDUS', 'CHAS', 'NOX' ,'RM', 'AGE', 'DIS' ,
 'B' ,'LSTAT']
'''
dataset = pd.DataFrame(dataset.data,columns=dataset.feature_names)

x = dataset[[ 'CRIM' ,'ZN' ,'INDUS', 'CHAS', 'NOX' ,'RM', 'AGE', 'DIS' ,
 'B' ,'LSTAT']]


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

model = GradientBoostingRegressor()
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
XGBRFRegressor
acc : 0.845630691400949
[0.04008065 0.00897015 0.03781021 0.00796106 0.0317596  0.3090608
 0.02043761 0.0986004  0.01736996 0.42794964]

DecisionTreeRegressor
acc : 0.7201287695179781
[2.72889982e-02 1.52677585e-03 1.41027112e-02 1.56534074e-04
 2.30956809e-02 2.57874327e-01 1.53603992e-02 8.91505518e-02
 1.13081509e-02 5.60135871e-01]

GradientBoostingRegressor
acc : 0.860751961247402
[3.35685385e-02 4.56677448e-04 1.00392592e-02 1.69735969e-03
 3.10336400e-02 3.15838503e-01 5.49624456e-03 1.04189584e-01
 1.11449476e-02 4.86535246e-01]

RandomForestRegressor
acc : 0.8451942292928873
[0.04531417 0.00219723 0.0135952  0.0015045  0.01637891 0.34234136
 0.01779811 0.07733001 0.01328635 0.47025416]

'''
