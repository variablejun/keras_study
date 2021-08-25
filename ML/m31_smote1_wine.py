from sklearn.datasets import load_wine
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')
dataset = load_wine()

x = dataset.data
y = dataset.target
print(x.shape, y.shape) #(178, 13) (178,)
print(pd.Series(y).value_counts())
'''
1    71
0    59
2    48
dtype: int64
'''
print(y)
'''
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
'''
x_new = x[:-30]#뒤에서부터
y_new = y[:-30]
print(x_new.shape, y_new.shape) #(148, 13) (148,)
print(pd.Series(y_new).value_counts())
'''
1    71
0    59
2    18
dtype: int64
'''
# train_size default 0.75
x_train, x_test, y_train, y_test = train_test_split(x_new,y_new,random_state=66,
shuffle=True,train_size=0.75,stratify=y_new) # y의 라벨의 비율이 동일하게 값은바뀌지만 개수는안바뀜
print(pd.Series(y_train).value_counts())
'''
1    53
0    44
2    14
전체 데이터를 증폭을하는것은 의미가 없다 (train 만 증폭)
'''

model = XGBClassifier(n_jobs=-1)

model.fit(x_train, y_train,eval_metric=['mlogloss'])
score = model.score(x_test,y_test,)

smote = SMOTE(random_state=66)
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
score2 = model2.score(x_test,y_test)

print('score 적용후: ',score2)
'''
2번을 30개 삭제한 데이터
score :  0.9459459459459459
smote 적용
score :  0.972972972972973
-20
-20
score 적용전:  0.975
score 적용후:  0.975
-10
-10
score 적용전:  0.9761904761904762
score 적용후:  0.9761904761904762
0.8
score 적용전:  0.9333333333333333
score 적용후:  0.9666666666666667
0.6
score 적용전:  0.9166666666666666
score 적용후:  0.9333333333333333
0.7
score 적용전:  0.9111111111111111
score 적용후:  0.9111111111111111
'''
print('smote 적용전',x_train.shape, y_train.shape)
print('smote 적용후',x_smote_tr.shape, y_smote_tr.shape)
print('smote 적용전 값 분포',pd.Series(y_train).value_counts())
print('smote 적용후 값 분포',pd.Series(y_smote_tr).value_counts())