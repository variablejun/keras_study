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

from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer

 
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)  # train에서 사용함
x_test = scaler.transform(x_test)


OE = OneHotEncoder()
y_test = y_test.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
OE.fit(y_test)
y_test = OE.transform(y_test).toarray() 
OE.fit(y_train)
y_train = OE.transform(y_train).toarray()


x = tf.placeholder(tf.float32,shape=[None,28*28])
y = tf.placeholder(tf.float32,shape=[None,10])

w1 = tf.Variable(tf.random_normal([28*28,256])) 
b1 = tf.Variable(tf.random_normal([256]))

layer1 = tf.nn.relu(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([256,128])) 
b2 = tf.Variable(tf.random_normal([128]))

layer2 = tf.nn.relu(tf.matmul(layer1, w2) + b2)

w3 = tf.Variable(tf.random_normal([128,64])) 
b3 = tf.Variable(tf.random_normal([64]))

layer3 = tf.nn.relu(tf.matmul(layer2, w3) + b3)

w4 = tf.Variable(tf.random_normal([64,32])) 
b4 = tf.Variable(tf.random_normal([32]))

layer4 = tf.nn.relu(tf.matmul(layer3, w4) + b4)

dropout1 = tf.nn.dropout(layer4,keep_prob=0.3)

w5 = tf.Variable(tf.random_normal([32,10])) 
b5 = tf.Variable(tf.random_normal([10]))

hypothesis = tf.nn.softmax(tf.matmul(dropout1, w5) + b5)
# hypothesis = tf.sigmoid(tf.matmul(layer4, w5) + b5)
# hypothesis = tf.nn.dropout(layer4,keep_prob=0.3) 드롭아웃
# cost = tf.reduce_mean(tf.square(hypothesis-y)) mse

cost = -tf.reduce_mean(y*tf.log(hypothesis) +( 1 -y) * tf.log(1-hypothesis)) #binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 훈련
for epochs in range(2001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_train,y:y_train})
     if epochs % 200 == 0:
          print(epochs,cost_val, '\n',hy_val)

# 예측
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted,acc], feed_dict={x:x_test,y:y_test})
print('예측값 \n',hy_val,'\n 결과값 : \n',c,'\n acc :',a)


'''
acc 0.97 이상


'''