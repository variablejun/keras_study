import numpy as np
from sklearn.decomposition import PCA

from tensorflow.keras.datasets import mnist

(x_train, _),(x_test, _) = mnist.load_data() #_ 무시하겟다 받지않겟다.
print(x_train.shape, x_test.shape) # (60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test,axis=0)
print(x.shape) # (70000, 28, 28)
pca = PCA(n_components=784) # 랜덤이 아닌 자기 기준에 따라서 압축을 시킨다 -> 제거하는것은 아님
x = x.reshape(70000,28*28)
x = pca.fit_transform(x)
print(x)
# 0.95 이상의 컴포넌트


pcaEVR = pca.explained_variance_ratio_
print(pcaEVR) # 
cunsum = np.cumsum(pcaEVR)
print(cunsum)
print(np.argmax(cunsum >= 0.95)+1) # 7

'''
0.95

154

'''
