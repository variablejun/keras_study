import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data # 442 10
y = dataset.target # 442,
pca = PCA(n_components=10) # 랜덤이 아닌 자기 기준에 따라서 압축을 시킨다 -> 제거하는것은 아님

x = pca.fit_transform(x)
pcaEVR = pca.explained_variance_ratio_
print(pcaEVR) # 
cunsum = np.cumsum(pcaEVR)
print(cunsum)
print(np.argmax(cunsum >= 0.94)+1) # 7

import matplotlib.pyplot as plt
plt.plot(cunsum)
plt.grid()
plt.show()
'''
몇개 까지 줄일것인지 차트로 확인하고 몇개가 가장 원데이터에 손실이 없는지 확인후 숫자를 넣어 
활용하는게 좋다.
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
 0.05365605] 
열 개수 만큼 주고 다 더하면  1
누적해서 더하는것 cunsum
[0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
 0.94794364]
수를 보고 주고싶은 만큼 n_com 조절해야한다
'''
'''
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.8, random_state=12,shuffle=True)



from xgboost import XGBRegressor

model = XGBRegressor()


model.fit(x_train,y_train)

result = model.score(x_test, y_test)

print(result)
'''
'''

0.95

0.1580171646245636

0.9
0.15545364003121764

'''