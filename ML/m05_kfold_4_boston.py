from sklearn.datasets import load_boston
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np 
dataset = load_boston()
import warnings
warnings.filterwarnings('ignore') # 워닝 무시

print(dataset.DESCR)
print(dataset.feature_names)

x = dataset.data
y = dataset.target


from sklearn.model_selection import train_test_split, KFold,cross_val_score

kfold = KFold(n_splits=5,random_state=76,shuffle=True)


from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
model = RandomForestRegressor()


score = cross_val_score(model,x,y,cv=kfold)
print('acc : ',score)
print("평균 : ",round(np.mean(score),4))

'''
LinearSVC
acc :  [nan nan nan nan nan]
평균 :  nan

SVC
acc :  [nan nan nan nan nan]
평균 :  nan

KNeighborsRegressor
acc :  [0.27761231 0.57263782 0.54127382 0.52007721 0.53480131]
평균 :  0.4893

LinearRegression
acc :  [0.64619546 0.78742655 0.69510789 0.5738149  0.73962297]
평균 :  0.6884

DecisionTreeRegressor
acc :  [0.68779525 0.82886153 0.75698149 0.80208932 0.83456234]
평균 :  0.7821

RandomForestClassifier
acc :  [0.81328505 0.86920405 0.83834592 0.87422592 0.90567997]
평균 :  0.8601
'''