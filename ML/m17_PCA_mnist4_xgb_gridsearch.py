# 0.95 xgb와 그리드서치 이용, ten DNN, CNN과 비교
params = [{'n_estimators':[100,200,300], 'leaning_rate':[0.1,0.3,0.001,0.02], 'max_depth':[4,5,6]},
          {'n_estimators':[90,100,110], 'leaning_rate':[0.1,0.001,0.01],
           'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.9,1]},
          {'n_estimators':[90,100], 'leaning_rate':[0.1,0.001,0.01], 
          'max_depth':[4,5,6], 'colsample_bytree':[0.6,0.9,1],'colsample_bylevel':[0.6,0.9,1]},
]
n_jobs = -1