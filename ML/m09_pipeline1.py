import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
from sklearn.metrics import r2_score,accuracy_score
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

dataset = load_iris()
print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


from sklearn.model_selection import train_test_split, KFold,cross_val_score
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.7, random_state=76)

kfold = KFold(n_splits=5,random_state=76,shuffle=True)

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression #  Regression은 다 회귀지만 유일하게 LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, RandomTreesEmbedding

from sklearn.pipeline import make_pipeline, Pipeline


model = make_pipeline(MinMaxScaler(),SVC())

score = cross_val_score(model,x_train,y_train,cv=kfold)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print(accuracy_score(y_test,y_pred)) 

'''
0.9555555555555556
'''