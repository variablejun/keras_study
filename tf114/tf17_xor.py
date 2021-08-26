import tensorflow as tf

tf.compat.v1.set_random_seed(77)

x_data = [[0,0],[0,1],[1,0],[1,1]] #4,2
y_data = [[0],[1],[1],[0]] # 4,1
x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([2,2])) 
b = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(x, w) + b)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) mse
cost = -tf.reduce_mean(y*tf.log(hypothesis) +( 1 -y) * tf.log(1-hypothesis)) #binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

# 훈련
for epochs in range(5001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_data,y:y_data})
     if epochs % 200 == 0:
          print(epochs,cost_val, '\n',hy_val)

# 예측
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
acc = tf.reduce_mean(tf.cast(tf.equal(predicted, y), dtype=tf.float32))

c, a = sess.run([predicted,acc], feed_dict={x:x_data,y:y_data})
print('예측값 \n',hy_val,'\n 결과값 : \n',c,'\n acc :', a)

'''

 결과값 :
 [[1. 1.]
 [1. 1.]
 [0. 1.]
 [0. 0.]]
 acc : 0.625
'''