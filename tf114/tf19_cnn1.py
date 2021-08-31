import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
tf.compat.v1.set_random_seed(77)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

y_test = to_categorical(y_test)
y_train = to_categorical(y_train)

learning_rate = 0.001
tr_epochs = 15
batch_size = 100
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32,shape=[None,28,28,1])
y = tf.placeholder(tf.float32,shape=[None,10])

w1 = tf.get_variable('w1',shape = [3,3,1,32]) # 초기값을 넣을 필요가없다
#3,3 = 커널사이즈 ,1 채널의 수(x에서 주는것) 32 필터 
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')


print(w1) #3, 3, 1, 32
print(L1) #?, 28, 28, 32
'''

model= Sequetial()
model.add(Conv2D(filter=32,kernel_size=(3,3),strides=1,padding='same',input_size=(28,28,1)))

'''