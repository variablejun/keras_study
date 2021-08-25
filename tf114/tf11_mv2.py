import tensorflow as tf

tf.compat.v1.set_random_seed(77)

x_data = [[73,51,65],[92,98,11],[89,31,33],[99,33,100],[17,66,79]] # 5,3

y_data = [[152],[185],[180],[205],[142]] # 5,1
x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32,shape=[None,1])

w = tf.Variable(tf.random_normal([3,1])) # 5,3 * 3,1 행렬곱을 하면 33없어지고 5,1이남는다
#행렬곱을 할때 쉐입이 맞아야 가능하다 맞지 않으면 곱할 수 없다.
#ValueError: Dimensions must be equal, but are 3 and 5 for 'MatMul' (op: 'MatMul') with input shapes: [?,3], [5,1].

b = tf.Variable(tf.random_normal([1]))

#hypothesis = x1 * w1  + x2 * w2 +  x3 * w3 + b
hypothesis = tf.matmul(x, w) + b
cost = tf.reduce_mean(tf.square(hypothesis-y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-7)
train = optimizer.minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x:x_data,y:y_data})
     if epochs % 10 == 0:
          print(epochs,cost_val, '\n',hy_val)


'''
y = [[152],[185],[180],[205],[142]]


2000 334.10797 
 [[168.66785]
 [187.94028]
 [143.74937]
 [206.8932 ]
 [133.85228]]

'''