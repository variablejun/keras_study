from xgboost import  XGBRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


dataset = load_boston()
x = dataset['data']
y = dataset['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.8, random_state=76)

scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


model = XGBRegressor(n_estimators=300, learning_rate=0.1,n_jobs=1)


model.fit(x_train,y_train,verbose=1,eval_metric=['rmse','mae','logloss'],
eval_set=[(x_train,y_train),(x_test,y_test)],
early_stopping_rounds=10)#앞은 훈련데이터 뒤에는 발리데이션, 돌아가는 모습을 보기위한것

result = model.score(x_test, y_test)
print(result)
y_pred  = model.predict(x_test)
r2 =r2_score(y_test,y_pred)
print(r2)
'''
learning_rate=0.001
-7.620252238041479
-7.620252238041479

 learning_rate=0.1
0.8297940837787662
0.8297940837787662
'''
import matplotlib.pyplot as plt

from xgboost import  XGBRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


dataset = load_boston()
x = dataset['data']
y = dataset['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.8, random_state=76)

scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


model = XGBRegressor(n_estimators=10000, learning_rate=0.001,
n_jobs=8,tree_method='gpu_hist',predictor='gpu_predictor') # cpu_predictor
# njobcpu의 코어를 몇개사용할 것인지 만다고 무조건 좋지 않음
import time
st = time.time()
model.fit(x_train,y_train,verbose=1,eval_metric=['rmse','mae','logloss'],
eval_set=[(x_train,y_train),(x_test,y_test)])#앞은 훈련데이터 뒤에는 발리데이션, 돌아가는 모습을 보기위한것
ed = time.time() - st

result = model.score(x_test, y_test)
print(result)
y_pred  = model.predict(x_test)
r2 =r2_score(y_test,y_pred)
print(r2)
print('걸린시간',ed)
'''
-1
걸린시간 13.095337390899658

njob 1
걸린시간 13.78065276145935

2
걸린시간 11.079406976699829

3
걸린시간 10.569080114364624

4
걸린시간 10.383572578430176

5
걸린시간 11.035303354263306

6
걸린시간 11.618136167526245

7
걸린시간 12.423393249511719

8
걸린시간 12.537104845046997

tree_method='gpu_hist' gpu로 연산시
걸린시간 56.2593936920166
gpu_id = 0 gpu가 여러개있을때 연산할 gpu선택

데이터의 따라서 GPU가 빠를때가 있고 CPU가 빠르기도하다
'''