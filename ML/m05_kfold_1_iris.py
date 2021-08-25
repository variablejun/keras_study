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
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것
model = LinearSVC()

score = cross_val_score(model,x_train,y_train,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))


'''
LinearSVC
acc :  [0.95238095 0.95238095 1.         1.         0.9047619 ]
평균 :  0.9619

SVC
acc :  [1.         1.         1.         0.95238095 0.9047619 ]
평균 :  0.9714

KNeighborsClassifier
acc :  [1.        1.        1.        0.9047619 0.9047619]
평균 :  0.9619

LogisticRegression
acc :  [1.         1.         1.         0.95238095 0.9047619 ]
평균 :  0.9714

DecisionTreeClassifier
acc :  [0.9047619  0.9047619  1.         0.9047619  0.95238095]
평균 :  0.9333

RandomForestClassifier
acc :  [0.95238095 0.9047619  1.         0.95238095 0.9047619 ]
평균 :  0.9429
'''