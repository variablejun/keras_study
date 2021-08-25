#이미지, 데이터 출력 완성 하시오 32,32,3

# 이미지 확인
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data() 
print(x_train.shape,y_train.shape )
print(x_test.shape,y_test.shape )
plt.imshow(x_train[11]) 
plt.show()
'''
(50000, 32, 32, 3) (50000, 1)
(10000, 32, 32, 3) (10000, 1)

'''