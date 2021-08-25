# 근접된 거리를 평균내서 점점 비슷한 특성값에 가까워지게하는것 최근접

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np  
import pandas as pd
dataset = load_iris()
print(dataset.target)
irisDF = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)
kmean = KMeans(n_clusters=3,max_iter=300,random_state=66) # max_iter= epoch n_clusters=생성할 y라벨값
kmean.fit(irisDF)

result = kmean.labels_
print(result)
'''
dataset.target
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
 
result = kmean.labels_ x값으로만 y값을 생성한것,
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 2 2 1 2 2 2 2
 2 2 1 1 2 2 2 2 1 2 1 2 1 2 2 1 1 2 2 2 2 2 1 2 2 2 2 1 2 2 2 1 2 2 2 1 2
 2 1]
'''
irisDF['cluster'] = kmean.labels_ # 클러스터로 생성한 y값
irisDF['target'] = dataset.target # 원 y값
print(dataset.feature_names) #['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

iris_result = irisDF.groupby(['target','cluster'])['sepal length (cm)'].count()
print(iris_result)
'''
0       0          50
1       1          48
        2           2
2       1          14 타겟이 2 일때 1인것 다른개수
        2          36
Name: sepal length (cm), dtype: int64
'''