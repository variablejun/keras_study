from sklearn.datasets import load_iris, load_boston,load_breast_cancer,load_diabetes,load_wine
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10,cifar100
import numpy as np
dataset = load_iris()
x_data = dataset.data
y_data = dataset.target
#'./_save/_npy/k55_x_data_iris.npy'
#'./_save/_npy/k55_y_data_iris.npy'
np.save('./_save/_npy/k55_x_data_iris.npy',arr=x_data)
np.save('./_save/_npy/k55_y_data_iris.npy',arr=y_data)
dataset = load_boston()
x_data = dataset.data
y_data = dataset.target
#'./_save/_npy/k55_x_data_boston.npy'
#'./_save/_npy/k55_y_data_boston.npy'
np.save('./_save/_npy/k55_x_data_boston.npy',arr=x_data)
np.save('./_save/_npy/k55_y_data_boston.npy',arr=y_data)
#'./_save/_npy/k55_x_data_breast_cancer.npy'
#'./_save/_npy/k55_y_data_breast_cancer.npy'
dataset = load_breast_cancer()
x_data = dataset.data
y_data = dataset.target
np.save('./_save/_npy/k55_x_data_breast_cancer.npy',arr=x_data)
np.save('./_save/_npy/k55_y_data_breast_cancer.npy',arr=y_data)
#'./_save/_npy/k55_x_data_diabetes.npy'
#'./_save/_npy/k55_y_data_diabetes.npy'
dataset = load_diabetes()
x_data = dataset.data
y_data = dataset.target
np.save('./_save/_npy/k55_x_data_diabetes.npy',arr=x_data)
np.save('./_save/_npy/k55_y_data_diabetes.npy',arr=y_data)
#'./_save/_npy/k55_x_data_wine.npy'
#'./_save/_npy/k55_y_data_wine.npy'

dataset = load_wine()
x_data = dataset.data
y_data = dataset.target
np.save('./_save/_npy/k55_x_data_wine.npy',arr=x_data)# 데이터 불러오는 시간을 줄이기 위해 넘파이로 저장하여 사용하도록
np.save('./_save/_npy/k55_y_data_wine.npy',arr=y_data)

#(x_train, y_train), (x_test, y_test) = mnist.load_data() 
(x_train, y_train), (x_test, y_test) = mnist.load_data()
np.save('./_save/_npy/k55_x_data_mnist_x_train.npy',arr=x_train)
np.save('./_save/_npy/k55_y_data_mnist_y_train.npy',arr=y_train)
np.save('./_save/_npy/k55_x_data_mnist_x_test.npy',arr=x_test)
np.save('./_save/_npy/k55_y_data_mnist_y_test.npy',arr=y_test)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
np.save('./_save/_npy/k55_x_data_fashion_mnist_x_train.npy',arr=x_train)
np.save('./_save/_npy/k55_y_data_fashion_mnist_y_train.npy',arr=y_train)
np.save('./_save/_npy/k55_x_data_fashion_mnist_x_test.npy',arr=x_test)
np.save('./_save/_npy/k55_y_data_fashion_mnist_y_test.npy',arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
np.save('./_save/_npy/k55_x_data_cifar10_x_train.npy',arr=x_train)
np.save('./_save/_npy/k55_y_data_cifar10_y_train.npy',arr=y_train)
np.save('./_save/_npy/k55_x_data_cifar10_x_test.npy',arr=x_test)
np.save('./_save/_npy/k55_y_data_cifar10_y_test.npy',arr=y_test)

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
np.save('./_save/_npy/k55_x_data_cifar100_x_train.npy',arr=x_train)
np.save('./_save/_npy/k55_y_data_cifar100_y_train.npy',arr=y_train)
np.save('./_save/_npy/k55_x_data_cifar100_x_test.npy',arr=x_test)
np.save('./_save/_npy/k55_y_data_cifar100_y_test.npy',arr=y_test)