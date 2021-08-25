import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target



from sklearn.model_selection import train_test_split, KFold,cross_val_score

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것
model = RandomForestClassifier()

score = cross_val_score(model,x,y,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))


'''
LinearSVC
acc :  [0.96666667 0.96666667 0.93333333 0.93333333 0.9       ]
평균 :  0.94

SVC
acc :  [0.93333333 0.96666667 1.         0.93333333 0.93333333]
평균 :  0.9533

KNeighborsClassifier
acc :  [0.93333333 0.96666667 1.         0.96666667 0.96666667]
평균 :  0.9667

LogisticRegression
acc :  [0.96666667 0.96666667 1.         0.93333333 0.93333333]
평균 :  0.96

DecisionTreeClassifier
acc :  [0.96666667 0.96666667 1.         0.9        0.9       ]
평균 :  0.9467

RandomForestClassifier
acc :  [0.93333333 0.96666667 1.         0.93333333 0.86666667]
평균 :  0.94

'''