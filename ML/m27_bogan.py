'''
결측치 처리
#1 행삭제
#2 0으로 처리 (특정값) 
#3앞에 값으로 채우기
#4뒤에값으로 채우기
#5중위값으로채우기
#6 bogan
7# 모델링 결측치를 빼고 가중치를 하고 predict를 할때 예측값에 결측치를 넣어줌
#9 부스트 계열 결측치에 자유롭다(트리계열) xg,dt,rf,lgbm 사용
'''
#[1,np.nan, np.nan, 9,10]
from pandas import DataFrame, Series
from datetime import datetime
import numpy as np
import pandas as pd

datastr = ['8/13/2021','8/14/2021','8/15/2021','8/16/2021','8/17/2021']
dates = pd.to_datetime(datastr)
ts = Series([1,np.nan, np.nan, 9,10],index=dates)
print(ts)
'''
2021-08-13     1.0
2021-08-14     NaN
2021-08-15     NaN
2021-08-16     9.0
2021-08-17    10.0
dtype: float64
'''
ts_intp_linear = ts.interpolate()
print(ts_intp_linear)
'''
2021-08-13     1.000000
2021-08-14     3.666667
2021-08-15     6.333333
2021-08-16     9.000000
2021-08-17    10.000000
dtype: float64
'''