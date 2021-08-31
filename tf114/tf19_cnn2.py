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

learning_rate = 0.01
tr_epochs = 3
batch_size = 32
total_batch = int(len(x_train)/batch_size)

x = tf.placeholder(tf.float32,shape=[None,28,28,1])
y = tf.placeholder(tf.float32,shape=[None,10])


#L1
w1 = tf.get_variable('w1',shape = [3,3,1,32]) # 초기값을 넣을 필요가없다
#3,3 = 커널사이즈 ,1 채널의 수(x에서 주는것) 32 필터(output) 
L1 = tf.nn.conv2d(x, w1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool = tf.nn.max_pool2d(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(w1) #3, 3, 1, 32
print(L1) #?, 28, 28, 32
print(L1_maxpool)
'''

model= Sequetial()
model.add(Conv2D(filter=32,kernel_size=(3,3),strides=1,padding='same',input_size=(28,28,1),activation='relu'))
model.add(Maxpool2D())
'''
#L2
w2 = tf.get_variable('w2',shape = [3,3,32,64])
L2 = tf.nn.conv2d(L1_maxpool, w2, strides=[1,1,1,1], padding='SAME')
L2 = tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool2d(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(w2) #3, 3, 32, 64
print(L2) #?, 14, 14, 64
print(L2_maxpool) #?, 7, 7, 64

#L3
w3 = tf.get_variable('w3',shape = [3,3,64,128])
L3 = tf.nn.conv2d(L2_maxpool, w3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.elu(L3)
L3_maxpool = tf.nn.max_pool2d(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(w3) #3, 3, 64, 128
print(L3) #?, 7, 7, 128
print(L3_maxpool) #?, 4, 4, 128

#L4
w4 = tf.get_variable('w4',shape = [2,2,128,64],initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3_maxpool, w4, strides=[1,1,1,1], padding='VALID')
L4 = tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool2d(L4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

print(w4) #2, 2, 128, 64
print(L4) #?, 3, 3, 64
print(L4_maxpool) #?, 2, 2, 64

#flatten
L_flat = tf.reshape(L4_maxpool,[-1,2*2*64])
print(L_flat) #?, 256
#L5 DNN
w5 = tf.get_variable('w5',shape = [2*2*64,64])
b1 = tf.Variable(tf.random_normal([64]),name = 'b1')
L5 = tf.matmul(L_flat, w5) +b1

L5 = tf.nn.relu(L5)
L5 = tf.nn.dropout(L5,keep_prob = 0.2)
print(L5) #?, 64


#L6 DNN
w6 = tf.get_variable('w6',shape = [64,32])
b2 = tf.Variable(tf.random_normal([32]),name = 'b2')
L6 = tf.matmul(L5, w6) +b2

L6 = tf.nn.relu(L6)
L6 = tf.nn.dropout(L6,keep_prob = 0.2)
print(L6) #?, 32

#L7 SOFTMAX
w7 = tf.get_variable('w7',shape = [32,10])
b3 = tf.Variable(tf.random_normal([10]),name = 'b3')
L7 = tf.matmul(L6, w7) +b3

hypothesis = tf.nn.softmax(L7)
print(hypothesis) #?, 10)


loss = tf.reduce_mean(-tf.reduce_min(y + tf.log(hypothesis),axis=1)) # 카테고리컬
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(tr_epochs):
     avg_loss = 0
     for i in range(total_batch):
          start = i * batch_size
          end = start * batch_size
          batch_x, batch_y = x_train[start:end], y_train[start:end]
          feed_dict = {x:batch_x,y:batch_y}
          batch_loss,_ = sess.run([loss,optimizer],feed_dict=feed_dict)
          avg_loss += batch_size/total_batch
     print('epoch:','%04d'% (epochs + 1),'loss:','{:.9f}',format(avg_loss))

prediction = tf.equal(tf.arg_max(hypothesis,1), tf.arg_max(y)) 
acc = tf.reduce_mean(tf.cast(prediction,tf.float32))

print(sess.run(acc,feed_dict={x:x_train,y:y_train}))