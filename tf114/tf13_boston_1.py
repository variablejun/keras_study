#실습 지표는 r2 
#최종결론값 r2
from sklearn.datasets import load_boston

import tensorflow as tf
# 506 13 506 ,
tf.compat.v1.set_random_seed(77)
dataset = load_boston()
x_data = dataset.data
y_data = dataset.target
# x_data = tf.cast(x_data,tf.float32)
# y_data = tf.cast(y_data,tf.float32)

y_data = y_data.reshape(-1,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
train_size = 0.01, random_state=66)

x = tf.placeholder(tf.float32,shape=[None,13])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([13,1])) 
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(5001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_train,y:y_train})
     if epochs % 200 == 0:
          print(epochs,cost_val)

predicted = sess.run(hypothesis,feed_dict={x:x_test})
from sklearn.metrics import r2_score
score = r2_score(y_test, predicted)

print(score)