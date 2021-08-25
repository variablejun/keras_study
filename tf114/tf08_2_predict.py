'''
실습
1, 4
2 56
3 678
predict 코드추가
x_test 라는 placeholder 생성
y =w_val * xtest + b_val
'''
# y = wx + b

import tensorflow as tf

tf.set_random_seed(99)

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])
x_test = tf.constant([6,7,8],tf.float32)
#x_test = tf.constant([5,6],tf.float32)
#x_test = tf.constant(4.0,tf.float32)

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
     x_pred = sess.run(w_val * x_test+ b_val)
     if step % 20 == 0:
          print(step,loss_val,w_val,b_val,x_pred)
     

print(x_pred)

'''
[4.002262]
[5.003572  6.0048823]
[6.0048823 7.006192  8.007502 ]
'''