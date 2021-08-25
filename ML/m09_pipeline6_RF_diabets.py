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
kfold = KFold(n_splits=5,random_state=76,shuffle=True)

'''


start = time.time()
model = make_pipeline(MinMaxScaler(),RandomForestRegressor())

model.fit(x_train,y_train)

from sklearn.metrics import r2_score,accuracy_score
y_pred = model.predict(x_test)

print('R2:',r2_score(y_test,y_pred)) # model.score(x_test,y_test) 와 같다
end = time.time() - start
print('걸린시간 : ', end)
'''
R2: 0.3597552523062928
걸린시간 :  0.154526948928833
'''