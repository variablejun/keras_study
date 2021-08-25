# y = wx + b

import tensorflow as tf

tf.set_random_seed(99)

x_train = [1,2,3]
y_train = [3,5,7]


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
     sess.run(train)
     if step % 20 == 0:
          print(step,sess.run(loss),sess.run(w),sess.run(b))