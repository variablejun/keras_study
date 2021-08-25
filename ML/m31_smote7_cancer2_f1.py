#라벨 0 112삭제
# 345 0 
# 6 1
# 789 2
#스모트 바뀌기전과 비교 더 좋아지게(F1)
from sklearn.datasets import load_breast_cancer
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import time
from sklearn.metrics import accuracy_score,f1_score
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()

x = dataset.data
y = dataset.target


print(x.shape, y.shape) #(4898, 11) (4898,)
print(pd.Series(y).value_counts())
'''
1    357
0    212

'''
print(y)

#라벨 통합
'''
0 1 밖에없음
for index, value in enumerate(y):
     if value == 3:
          y[index] = 0
     elif value == 4:
          y[index] = 0
     elif value == 5 :
          y[index] = 0 
     elif value == 6 :
          y[index] = 1  
     elif value == 7 :
          y[index] = 2 
     elif value == 8 :
          y[index] = 2 
     elif value == 9 :
          y[index] = 2 
'''            
     

print(pd.Series(y).value_counts())
x_new = x[:-112]#뒤에서부터
y_new = y[:-112]
# train_size default 0.75
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=66,
shuffle=True,train_size=0.75) # y의 라벨의 비율이 동일하게 값은바뀌지만 개수는안바뀜
print(pd.Series(y_train).value_counts())

'''
6.0    1648
5.0    1093
7.0     660
8.0     131
4.0     122
3.0      15
9.0       4
전체 데이터를 증폭을하는것은 의미가 없다 (train 만 증폭)
'''


model = XGBClassifier(n_jobs=-1)
start = time.time()
model.fit(x_train, y_train,eval_metric=['mlogloss'])
score = model.score(x_test,y_test)

y_pred = model.predict(x_test)
f1 = f1_score(y_test,y_pred, average='macro')
smote = SMOTE(random_state=66,k_neighbors=60) # smote의 기본값은 5개 이지만 지금 라벨 9의 개수는 4개이기 때문에 적용시킬수없다.
#ValueError: Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 6
#그래서 엮이는 노드수를 줄여서 돌아가게만든다. 연산수가 떨어져 성능이줄어들수있다
x_smote_tr, y_smote_tr = smote.fit_resample(x_train, y_train)

print(pd.Series(y_smote_tr).value_counts())

'''
0    53
1    53
2    53
'''

print(x_smote_tr.shape, y_smote_tr.shape) #(159, 13) (159,)
print('score 적용전: ',score)

model2 = XGBClassifier(n_jobs=-1)

model2.fit(x_smote_tr,y_smote_tr,eval_metric=['mlogloss'])
y_pred2 = model2.predict(x_test)
score2 = model2.score(x_test,y_test)
f2 = f1_score(y_test,y_pred2, average='macro')
print('score 적용후: ',score2)

print('smote 적용전',x_train.shape, y_train.shape)
print('smote 적용후',x_smote_tr.shape, y_smote_tr.shape)
print('smote 적용전 값 분포\n',pd.Series(y_train).value_counts())
print('smote 적용후 값 분포\n',pd.Series(y_smote_tr).value_counts())
end = time.time() -start
print('시간',end)
print('적용전 F1 :', f1 )
print('적용후 F1 :', f2 )

'''


k_neighbors=9
score 적용전:  0.9790209790209791
score 적용후:  0.986013986013986

적용전 F1 : 0.9770404067433769
적용후 F1 : 0.9847612958226768

k_neighbors=60
score 적용전:  0.9790209790209791
score 적용후:  0.993006993006993
적용전 F1 : 0.9770404067433769
적용후 F1 : 0.9923468022477924
'''