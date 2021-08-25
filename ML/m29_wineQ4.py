import numpy as np
import pandas as pd
from xgboost import XGBClassifier
datasets = pd.read_csv('../_data/winequality-white.csv', sep=';' , index_col=None, header=0)

datasets = datasets.values

x = datasets[:,:11]
y = datasets[:,11]

listQ = []

for i in list(y):
     if i <= 4:
          listQ += [0]
     elif i <= 7:
          listQ += [1]
     else:
          listQ += [2]
y = np.array(listQ)  # 

'''
pan -> num
values to_numpy
y의 라벨값을 3~9까지 퍼져있는걸 줄이는것
데이터가 줄어들지는 않는다
단 내가 라벨값을 줄일수있는지 파악해야함
'''


from sklearn.preprocessing import RobustScaler,StandardScaler


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=66,shuffle=True,train_size=0.8)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = XGBClassifier(n_jobs=-1)

model.fit(x_train,y_train)
score  = model.score(x_test,y_test)
print(score)# 0.9469387755102041