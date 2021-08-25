from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
dataset = load_breast_cancer()
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
model = LinearSVC()


score = cross_val_score(model,x,y,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))

'''
LinearSVC
acc :  [0.90350877 0.90350877 0.92105263 0.92105263 0.9380531 ]
평균 :  0.9174

SVC
acc :  [0.9122807  0.9122807  0.9122807  0.92105263 0.90265487]
평균 :  0.9121

KNeighborsClassifier
acc :  [0.92982456 0.94736842 0.95614035 0.9122807  0.92920354]
평균 :  0.935

LogisticRegression
acc :  [0.92105263 0.95614035 0.96491228 0.93859649 0.9380531 ]
평균 :  0.9438

DecisionTreeClassifier
acc :  [0.94736842 0.92982456 0.94736842 0.96491228 0.91150442]
평균 :  0.9402

RandomForestClassifier
acc :  [0.97368421 0.98245614 0.97368421 0.97368421 0.9380531 ]
평균 :  0.9683
'''