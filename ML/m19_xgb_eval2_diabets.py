from xgboost import  XGBRegressor
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score


dataset = load_diabetes()
x = dataset['data']
y = dataset['target']


x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.8, random_state=76)

scaler = StandardScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


model = XGBRegressor(n_estimators=300, learning_rate=0.1,n_jobs=1)


model.fit(x_train,y_train,verbose=1,eval_metric=['rmse','mae','logloss'],eval_set=[(x_train,y_train),(x_test,y_test)])#앞은 훈련데이터 뒤에는 발리데이션, 돌아가는 모습을 보기위한것

result = model.score(x_test, y_test)
print(result)
y_pred  = model.predict(x_test)
r2 =r2_score(y_test,y_pred)
print(r2)
'''
learning_rate=0.001
-2.4128504670890174
-2.4128504670890174

 learning_rate=0.1
0.22452675518188614
0.22452675518188614
'''