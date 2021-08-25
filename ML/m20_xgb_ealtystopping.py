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


model = XGBRegressor(n_estimators=300, learning_rate=0.1,n_jobs=1)


model.fit(x_train,y_train,verbose=1,eval_metric=['rmse','mae','logloss'],eval_set=[(x_train,y_train),(x_test,y_test)])#앞은 훈련데이터 뒤에는 발리데이션, 돌아가는 모습을 보기위한것

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

def plot_feature_importances_dataset(model):
     n_feature = dataset.data.shape[1]
     plt.barh(np.arange(n_feature),model.feature_importances_,align='center')
     plt.yticks(np.arange(n_feature),dataset.feature_names)
     plt.xlabel('feature_importances')
     plt.ylabel('features')
     plt.ylim(-1,n_feature)

plot_feature_importances_dataset(model)   
plt.show()