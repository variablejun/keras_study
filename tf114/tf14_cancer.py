from sklearn.datasets import load_breast_cancer

import tensorflow as tf
# 569 30 569 ,
tf.compat.v1.set_random_seed(77)
dataset = load_breast_cancer()
x_data = dataset.data
y_data =dataset.target
y_data = y_data.reshape(-1,1)

x = tf.placeholder(tf.float32,shape=[None,30])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([30,1])) 
b = tf.Variable(tf.random_normal([1]))

# 최종결과 acc ,평가예측까지
# predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
# acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
train_size = 0.2, random_state=66)

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) mse
cost = -tf.reduce_mean(y*tf.log(hypothesis) +( 1 -y) * tf.log(1-hypothesis)) #binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 훈련
for epochs in range(5001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_train,y:y_train})
     if epochs % 200 == 0:
          print(epochs,cost_val, '\n',hy_val)

# 예측
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))
c, a = sess.run([predicted,acc], feed_dict={x:x_test,y:y_test})
print('ac :', a)
'''

ac : 0.37280703
'''