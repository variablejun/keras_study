import tensorflow as tf

tf.compat.v1.set_random_seed(77)


x1_data = [73.,93.,89.,96.,73.] #국
x2_data = [80.,88.,91.,98.,66.] #영
x3_data = [75.,93.,90.,100.,70.] #수
y_data = [152.,185.,180.,196.,142.] #점수

x1 = tf.placeholder(tf.float32)
x2 = tf.placeholder(tf.float32)
x3 = tf.placeholder(tf.float32)

y = tf.placeholder(tf.float32)

w1 = tf.Variable(tf.random_normal([1]),name='weght1')
w2 = tf.Variable(tf.random_normal([1]),name='weght2')
w3 = tf.Variable(tf.random_normal([1]),name='weght3')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = x1 * w1  + x2 * w2 +  x3 * w3 + b

cost = tf.reduce_mean(tf.square(hypothesis-y))
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.00001)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(2001):
     cost_val, hy_val, _ = sess.run([cost,hypothesis,train], feed_dict={x1:x1_data,x2:x2_data,x3:x3_data,y:y_data})
     if epochs % 10 == 0:
          print(epochs,cost_val,hy_val)
          
     