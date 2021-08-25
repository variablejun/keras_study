import numpy as np

aaa = np.array([[1,2,10000,4,5,6,7,8,90,5000],
             [1000,2000,3,4000,5000,6000,7000,8,9000,1001]])
# 2,10

aaa = aaa.transpose()#10,2

from sklearn.covariance import EllipticEnvelope
outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

result = outliers.predict(aaa)
print(result)
'''
[ 1  1 -1  1  1  1  1  1  1 -1] 두 배열의 아웃라이어지점
EllipticEnvelope 키워드,파라미타 정리
'''