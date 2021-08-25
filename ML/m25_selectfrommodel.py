from xgboost import XGBRegressor
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel # 모델에서 피처를 선택
from sklearn.metrics import r2_score
#1 data
x, y = load_boston(return_X_y=True)

x_train, x_test,y_train, y_test = train_test_split(x,y,train_size=0.8,shuffle=True)
#2 model
model = XGBRegressor(n_jobs=8)

#3 fit
model.fit(x_train,y_train)

#4 pred
score = model.score(x_test,y_test)
print('score ', score)
threshold= np.sort(model.feature_importances_)

print(threshold)
'''
[0.00133304 0.00339307 0.00857026 0.01093002 0.01113533 0.01168353
 0.01923767 0.03113616 0.0328261  0.04722729 0.05724799 0.15774222
 0.6075373 ]
0.00133304 13 80.90273402378519
0.0033930738 12 81.34309434011442
0.008570257 11 81.28921956882357
0.0109300185 10 79.3423809971888
0.011135326 9 81.28917385061628
0.011683535 8 82.3059629220525
0.019237675 7 79.00196768939149
0.031136155 6 78.20469482192524
0.032826103 5 73.30851609211983
0.04722729 4 73.02168304240959
0.057247985 3 75.89884506112111
0.15774222 2 68.15291642211326
0.6075373 1 42.70769062439166
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
score  0.910131606886159

'''