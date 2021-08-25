import numpy as np

aaa = np.array([1,2,-1000,4,5,6,7,8,90,100,500])
'''
이상치를 구할때 평균에 2배를 하여 구하는데 평균값이 너무 크거나 작아질 수 있다
그래서 사분위수를 구해서 1사분위(q1)와 3사분위(q3)를 빼고 그 범위에 위아래로 (1~1.5배) 더한 범위를 벗어난것을 아웃라이어라고한다.
'''

def outliers(data_out):
     q1, q2, q3 = np.percentile(data_out,[25,50,75])
     print(q1)
     print(q2)
     print(q3)
     iqr = q3 - q1
     low = q1 - (iqr *1.5)
     high = q3 + (iqr *1.5)
     return np.where((data_out>high) | (data_out<low))

out_loc = outliers(aaa)
print('이상치 위치', out_loc)
'''
딱 떨어지는 위치가 아닐수 있다 백분위수의 25%에 위치에있는것
3.0
6.0
49.0
이상치 위치 (array([ 2, 10], dtype=int64),)

이상치 처리
1, 삭제
2, nan 처리후 -> 보간 // linear
3, 결측치와 유사
4, scaler rubs,qeunt
5, 모델링 트리계열 xg,dt,rf,lgbm 사용
'''
# 시각화 위 데이타 boxplot
import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()