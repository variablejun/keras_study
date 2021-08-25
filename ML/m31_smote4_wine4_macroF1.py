from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import numpy as np
import warnings
import time
from sklearn.metrics import accuracy_score,f1_score
warnings.filterwarnings('ignore')
dataset = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0)

dataset = dataset.values

x = dataset[:,:11]
y = dataset[:,11]


print(x.shape, y.shape) #(4898, 11) (4898,)
print(pd.Series(y).value_counts())
'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''
print(y)

#라벨 통합

for index, value in enumerate(y):
     if value == 9:
          y[index] = 7
     elif value == 8:
          y[index] = 7
     elif value == 3 :
          y[index] = 5 
     elif value == 4 :
          y[index] = 7          
     

print(pd.Series(y).value_counts())

# train_size default 0.75
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=66,
shuffle=True,train_size=0.75,stratify=y) # y의 라벨의 비율이 동일하게 값은바뀌지만 개수는안바뀜
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
smote = SMOTE(random_state=66)#,k_neighbors=3) # smote의 기본값은 5개 이지만 지금 라벨 9의 개수는 4개이기 때문에 적용시킬수없다.
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
print('smote 적용전 값 분포',pd.Series(y_train).value_counts())
print('smote 적용후 값 분포',pd.Series(y_smote_tr).value_counts())
end = time.time() -start
print('시간',end)
print('적용전 F1 :', f1 )
print('적용후 F1 :', f2 )

'''
적용전 F1 : 0.6847810535365474
적용후 F1 : 0.691649234547523
'''