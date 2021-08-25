import numpy as np 
from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

dataset = load_boston()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target



from sklearn.model_selection import train_test_split, KFold,cross_val_score,GridSearchCV,RandomizedSearchCV
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding
# 여러가지 트리연산들을 합치는 것
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.pipeline import make_pipeline, Pipeline
'''
params = [{'randomforestregressor__n_estimators':[100,200],'randomforestregressor__min_samples_leaf' : [3,5,7,10],'randomforestregressor__max_depth' : [6,8,10,12],'randomforestregressor__min_samples_split' : [2,3,5,10]},
        
          {'randomforestregressor__min_samples_leaf' : [3,5,7,10],'randomforestregressor__min_samples_split' : [2,3,5,10]},
          {'randomforestregressor__min_samples_split' : [2,3,5,10]}
     
]# __ 언더바 두개 모델과 파라미터 구분

'''
params = [{'rf__n_estimators':[100,200],'rf__min_samples_leaf' : [3,5,7,10]},
          {'rf__max_depth' : [6,8,10,12]},
          {'rf__min_samples_leaf' : [3,5,7,10],'rf__min_samples_split' : [2,3,5,10]},
          {'rf__min_samples_split' : [2,3,5,10]},
          {'rf__n_jobs' : [-1,2,4]}
]

pipe = Pipeline([("scaler",MinMaxScaler()),('rf',RandomForestRegressor())]) # 둘 차이 비교 다른것도 마찬가지


kfold = KFold(n_splits=5,random_state=76,shuffle=True)
#pipe = make_pipeline(MinMaxScaler(),RandomForestRegressor())
start = time.time()
model = GridSearchCV(pipe,params,cv=kfold,verbose=1)

model.fit(x_train,y_train)

print('acc : ',model.best_score_)
print(model.best_params_)
print("최적의 매개변수 ",model.best_estimator_)

end = time.time() - start
print('걸린시간 : ', end)
'''
make_pipeline
acc :  0.8645081474075766
{'randomforestregressor__min_samples_split': 2}
최적의 매개변수  Pipeline(steps=[('minmaxscaler', MinMaxScaler()),
                ('randomforestregressor', RandomForestRegressor())])
걸린시간 :  124.38881206512451

Pipeline
acc :  0.8677088547567223
{'rf__n_jobs': -1}
최적의 매개변수  Pipeline(steps=[('scaler', MinMaxScaler()),
                ('rf', RandomForestRegressor(n_jobs=-1))])
걸린시간 :  26.909996271133423

'''