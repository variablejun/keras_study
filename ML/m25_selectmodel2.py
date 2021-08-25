# 실습 그리드,랜덤서치로 튜닝한 모델 구성 최적의 R2값과 피처 임포턴스 구성
# 데이터는 디아벳
# selectmodel로 최적의 피처 임포턴스 구하기

# 피처개수를 줄이고 다시 튜닝해서 최적의 R2를 구하고 바교 0.47이상
from xgboost import XGBRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel # 모델에서 피처를 선택
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')
#1 data
x, y = load_diabetes(return_X_y=True)

x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True)
#2 model
params = [{'n_estimators':[100,200,500],'max_depth' : [6,8,10,12,24],'n_jobs' : [-1,2,4]},
          {'max_depth' : [6,8,10,12,25]},
          {'n_jobs' : [-1,2,4]}
]

model = RandomizedSearchCV(XGBRegressor(),params)

#3 fit
model.fit(x_train,y_train)

print('acc : ',model.best_score_)
print("최적의 매개변수 ",model.best_estimator_)
#4 pred
score = model.score(x_test,y_test)
print('score ', score)
#threshold= np.sort(model.feature_importances_)

#print(threshold)
'''

'''
'''
for i in threshold:
     selection = SelectFromModel(model,threshold=i,prefit=True)

     select_x_train = selection.transform(x_train)
     select_x_test = selection.transform(x_test)
     select_model = XGBRegressor(n_jobs=-1)
     select_model.fit(select_x_train,y_train)
     pred = select_model.predict(select_x_test)
     score = r2_score(y_test,pred)
     print(i, select_x_train.shape[1],score*100)
'''
'''
score  0.36772150444453977

'''