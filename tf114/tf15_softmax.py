import tensorflow as tf
import numpy as np
tf.compat.v1.set_random_seed(77)

x_data = [[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,6,7]]# 8,4
y_data = [[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]] # 8,3

x = tf.placeholder(tf.float32,shape=[None,4])
y = tf.placeholder(tf.float32,shape=[None,3])

w = tf.Variable(tf.random_normal([4,3])) # y 와 연결되어야 해서 3
b = tf.Variable(tf.random_normal([1,3])) # 바이어스는 입력은 1이지만 나갈때는 3개로 나감

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

loss = tf.reduce_mean(-tf.reduce_min(y + tf.log(hypothesis),axis=1)) # 카테고리컬
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001).minimize(loss) # 그라디언트 디센트방식으로 최소화

with tf.Session() as sess:

     sess.run(tf.global_variables_initializer())

     for epochs in range(2001):
          hy_val, loss_val = sess.run([loss,optimizer], feed_dict={x:x_data,y:y_data})
          if epochs % 200 == 0:
               print(epochs,'\n',loss_val, '\n',hy_val)

     result  = sess.run(hypothesis,feed_dict={x:[[1,11,7,9]]})
     print(result, sess.run(tf.argmax(result,1)))
     
'''
[[0.3729176  0.31853315 0.30854926]] [0]
'''
