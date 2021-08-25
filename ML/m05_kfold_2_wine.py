from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
dataset = load_wine()
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
model = RandomForestClassifier()


score = cross_val_score(model,x,y,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))

'''
LinearSVC
acc :  [0.91666667 0.83333333 0.97222222 0.85714286 0.8       ]
평균 :  0.8759

SVC
acc :  [0.66666667 0.61111111 0.72222222 0.68571429 0.65714286]
평균 :  0.6686

KNeighborsClassifier
acc :  [0.66666667 0.75       0.77777778 0.62857143 0.65714286]
평균 :  0.696

LogisticRegression
acc :  [0.91666667 0.94444444 0.97222222 0.97142857 0.91428571]
평균 :  0.9438

DecisionTreeClassifier
acc :  [0.94444444 0.91666667 0.86111111 0.88571429 0.97142857]
평균 :  0.9159

RandomForestClassifier
acc :  [0.97222222 1.         1.         0.94285714 1.        ]
평균 :  0.983

'''