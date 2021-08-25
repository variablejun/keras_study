# y = wx + b

import tensorflow as tf

tf.set_random_seed(99)

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
# random_normal 정규분포의 의한 랜덤값
hypothesis = x_train * w + b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
     _, loss_val, w_val, b_val = sess.run([train,loss,w,b],feed_dict={x_train:[1,2,3],y_train:[1,2,3]})
     if step % 20 == 0:
          print(step,loss_val,w_val,b_val)