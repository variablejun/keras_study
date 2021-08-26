import tensorflow as tf
from tensorflow.keras.datasets import mnist
tf.compat.v1.set_random_seed(77)

(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
'''
(60000, 28, 28) (60000,)
(10000, 28, 28) (10000,)
'''

from sklearn.preprocessing import OneHotEncoder
OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() 
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()


x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])

w = tf.Variable(tf.random_normal([28*28,10])) 
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) mse
cost = -tf.reduce_mean(y*tf.log(hypothesis) +( 1 -y) * tf.log(1-hypothesis)) #binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 훈련
for epochs in range(1001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_train,y:y_train})
     if epochs % 200 == 0:
          print(epochs,cost_val, '\n',hy_val)

# 예측
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted,acc], feed_dict={x:x_test,y:y_test})
print('예측값 \n',hy_val,'\n 결과값 : \n',c,'\n acc :', a)


'''
learning_rate=0.001
 결과값 :
 [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
 acc : 0.9

learning_rate=0.01
결과값 :
 [[0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 ...
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]
 [0. 0. 0. ... 0. 0. 0.]]
 acc : 0.9


'''