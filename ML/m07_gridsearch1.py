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



from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것

params = [{"C":[1,10,100,1000],'kernel':['linear']},
          {"C":[1,10,100],'kernel':['rbf'],'gamma':[0.001,0.0001]},
          {"C":[1,10,100,1000],'kernel':['sigmoid'],'gamma':[0.001,0.0001]}
]
model = GridSearchCV(SVC(),params,cv=kfold)

model.fit(x_train,y_train)
print('acc : ',model.best_score_)
print("최적의 매개변수 ",model.best_estimator_)
from sklearn.metrics import r2_score,accuracy_score
y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred)) # model.score(x_test,y_test) 와 같다
'''
acc :  0.9904761904761905
최적의 매개변수  SVC(C=1, kernel='linear')
0.9555555555555556
'''