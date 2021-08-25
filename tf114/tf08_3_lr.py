'''
08_2 파일의 epochs를 100번이하로 줄여라
결과치는 w=1.999999 b 0.999999
'''
import tensorflow as tf

tf.set_random_seed(99)

x_train = tf.placeholder(tf.float32,shape=[None])
y_train = tf.placeholder(tf.float32,shape=[None])
x_test = tf.constant(4.0,tf.float32)
#x_test = tf.constant([5,6],tf.float32)
#x_test = tf.constant(4.0,tf.float32)

w = tf.Variable(tf.random_normal([1]),dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]),dtype=tf.float32)

# random_normal 정규분포의 의한 랜덤값
hypothesis = x_train * w + b


loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.677)
train = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(100):
     
     _, loss_val, w_val, b_val = sess.run([train,loss,w,b],feed_dict={x_train:[1,2,3],y_train:[1,2,3]})
     x_pred = sess.run(w_val * x_test+ b_val)
     if step % 100 == 0:
          print(step,loss_val,w_val,b_val)
     
#결과치는 w=1.999999 b 0.999999
print(x_pred)

'''
0 1.4201835 [4.130016] [1.0095398]
[nan]
'''