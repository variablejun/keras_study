import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA

dataset = load_diabetes()
x = dataset.data # 442 10
y = dataset.target # 442,
pca = PCA(n_components=7) # 랜덤이 아닌 자기 기준에 따라서 압축을 시킨다 -> 제거하는것은 아님

x = pca.fit_transform(x)
print(x.shape) # 442,7
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,
train_size = 0.8, random_state=12,shuffle=True)



from xgboost import XGBRegressor

model = XGBRegressor()


model.fit(x_train,y_train)

result = model.score(x_test, y_test)

print(result)

'''
차원분리전
(442, 10)
0.2316559406166574

차원분리후
(442, 7)
0.15545364003121764
'''