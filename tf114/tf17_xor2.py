import tensorflow as tf

tf.compat.v1.set_random_seed(77)

x_data = [[0,0],[0,1],[1,0],[1,1]] #4,2
y_data = [[0],[1],[1],[0]] # 4,1

x = tf.placeholder(tf.float32,shape=[None,2])
y = tf.placeholder(tf.float32,shape=[None,1])
# 히든레이어
w1 = tf.Variable(tf.random_normal([2,5])) 
b1 = tf.Variable(tf.random_normal([5]))

layer1 = tf.sigmoid(tf.matmul(x, w1) + b1)

w2 = tf.Variable(tf.random_normal([5,3])) 
b2 = tf.Variable(tf.random_normal([3]))

layer2 = tf.sigmoid(tf.matmul(layer1, w2) + b2)

w4 = tf.Variable(tf.random_normal([3,10])) 
b4 = tf.Variable(tf.random_normal([10]))

layer4 = tf.sigmoid(tf.matmul(layer2, w4) + b4)

# 아웃풋레이어 앞뒤 앞뒤 로 생각하면 편하다. 히든 레이어에서 x를 연산한 값을 그 다음 레이어에 주고 계속해서 반복해서 딥러닝시킨다.
# 제귀함수를 생각하면 편하다 쉐입은 앞뒤앞뒤
w3 = tf.Variable(tf.random_normal([10,1])) 
b3 = tf.Variable(tf.random_normal([1]))

hypothesis = tf.sigmoid(tf.matmul(layer4, w3) + b3)

# cost = tf.reduce_mean(tf.square(hypothesis-y)) mse
cost = -tf.reduce_mean(y*tf.log(hypothesis) +( 1 -y) * tf.log(1-hypothesis)) #binary_crossentropy
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1)
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
learning_rate=1

 결과값 :
 [[0.]
 [1.]
 [1.]
 [0.]]
 acc : 1.0
 
'''