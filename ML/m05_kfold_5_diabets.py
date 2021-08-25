from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
dataset = load_diabetes()
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


from sklearn.model_selection import train_test_split, KFold,cross_val_score

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = SVC()


score = cross_val_score(model,x,y,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))

'''
LinearSVC
acc :  [0.         0.02247191 0.01136364 0.01136364 0.        ]
평균 :  0.009

SVC
acc :  [0.01123596 0.01123596 0.01136364 0.02272727 0.        ]
평균 :  0.0113

KNeighborsRegressor
acc :  [0.17494211 0.49105434 0.17627351 0.5075087  0.41623294]
평균 :  0.3532

LinearRegression
acc :  [0.4290448  0.60918943 0.28012882 0.58899164 0.4321181 ]
평균 :  0.4679

DecisionTreeRegressor
acc :  [-0.31443278 -0.04088632 -0.75917499  0.09097066 -0.14759228]
평균 :  -0.2342

RandomForestClassifier
acc :  [0.3000182  0.49933211 0.11053636 0.54600359 0.3943401 ]
평균 :  0.37
'''