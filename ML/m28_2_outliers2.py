import numpy as np

aaa = np.array([[1,2,10000,4,5,6,7,8,90,5000],
             [1000,2000,3,4000,5000,6000,7000,8,9000,1001],
             [1564,2440,300,400,530,775,1224,120,9000,1991],
             [10,222,3,1100,440,330,800,8,90,13301],
             ])
# 2,10

aaa = aaa.transpose()#10,2
print(aaa.shape)#(10, 4)

def outliers(data_out):
     listq=[]

     for i in range(data_out.shape[1]):
          q1, q2, q3 = np.percentile(data_out[:,i],[25,50,75])
          print(i+1,'번째 배열')
          print('1사분위',q1)
          print('2사분위',q2)
          print('3사분위',q3)
          
          iqr = q3 - q1
          low = q1 - (iqr *1.5)
          high = q3 + (iqr *1.5)
          a = np.where((data_out[:,i]>high) | (data_out[:,i]<low))
          listq.append(a)


out_loc = outliers(aaa)
print('이상치 위치', out_loc)
'''
이상치의 위치를 정확히 뽑아낼수가없다.
5.5
550.0
4250.0
이상치 위치 (array([], dtype=int64), array([], dtype=int64))
다차원의 이상치가 출력되도록 함수수정

import matplotlib.pyplot as plt
plt.boxplot(aaa)
plt.show()
'''
