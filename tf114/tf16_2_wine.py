# 기존 코드에 acc 추가
import tensorflow as tf
import numpy as np
from sklearn.datasets import load_wine
tf.compat.v1.set_random_seed(77)
dataset = load_wine()
x_data = dataset.data
y_data =dataset.target


print(x_data.shape, y_data.shape) #(178, 13) (178,)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
train_size = 0.6, random_state=77)


from sklearn.preprocessing import OneHotEncoder,StandardScaler,RobustScaler
scaler = RobustScaler()
scaler.fit(x_train) 
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)

OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() 
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
'''
(106, 13) (106, 3)
(72, 13) (72, 3)
'''

x = tf.placeholder(tf.float32,shape=[None,13])
y = tf.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random_normal([13,3])) # y 와 연결되어야 해서 3
b = tf.Variable(tf.random_normal([1,3])) # 바이어스는 입력은 1이지만 나갈때는 3개로 나감

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_min(y + tf.log(hypothesis),axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.011).minimize(loss) # 그라디언트 디센트방식으로 최소화

with tf.Session() as sess:

     sess.run(tf.global_variables_initializer())

     for epochs in range(2001):
          hy_val, loss_val = sess.run([loss,optimizer], feed_dict={x:x_train,y:y_train})
          if epochs % 200 == 0:
               print(epochs,'\n',loss_val, '\n',hy_val)

     predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
     acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

     c, a = sess.run([predicted,acc], feed_dict={x:x_test,y:y_test})
     print('예측값 \n',hy_val,'\n 결과값 : \n',c,'\n acc :', a)

'''
acc : 0.6527778
'''