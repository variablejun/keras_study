import numpy as np 
from sklearn.datasets import load_diabetes
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

dataset = load_diabetes()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target



from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것
import time
params = [{'n_estimators':[100,200],'min_samples_leaf' : [3,5,7,10],'max_depth' : [6,8,10,12],'min_samples_split' : [2,3,5,10],'n_jobs' : [-1,2,4]},
          {'max_depth' : [6,8,10,12]},
          {'min_samples_leaf' : [3,5,7,10],'min_samples_split' : [2,3,5,10]},
          {'min_samples_split' : [2,3,5,10]},
          {'n_jobs' : [-1,2,4]}
]
start = time.time()
model = RandomizedSearchCV(RandomForestRegressor(),params,cv=kfold,verbose=1)

model.fit(x_train,y_train)
print('acc : ',model.best_score_)
print(model.best_params_)
print("최적의 매개변수 ",model.best_estimator_)
from sklearn.metrics import r2_score,accuracy_score
y_pred = model.predict(x_test)


print(r2_score(y_test,y_pred)) # model.score(x_test,y_test) 와 같다
end = time.time() - start
print('걸린시간 : ', end)
'''

Fitting 5 folds for each of 411 candidates, totalling 2055 fits
acc :  0.41864252573753474
최적의 매개변수  RandomForestRegressor(max_depth=8, min_samples_leaf=5, min_samples_split=10,
                      n_jobs=-1)
0.42084821863710964
걸린시간 :  666.9764559268951

Fitting 5 folds for each of 10 candidates, totalling 50 fits
acc :  0.41247448540077986
최적의 매개변수  RandomForestRegressor(max_depth=12, min_samples_leaf=7, n_jobs=4)
0.4360939539989588
걸린시간 :  14.515647888183594
'''