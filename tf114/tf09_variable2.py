import tensorflow as tf

tf.compat.v1.set_random_seed(77)
x = [1,2,3]
w = tf.Variable([0.3],tf.float32)
b= tf.Variable([1.0],tf.float32)

hypothesis = x * w + b

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(hypothesis)
print(aaa) # [1.3       1.6       1.9000001]
sess.close

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = hypothesis.eval()
print(bbb) # [1.3       1.6       1.9000001]
sess.close

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
ccc = hypothesis.eval(session=sess) # [1.3       1.6       1.9000001]
print(ccc)
sess.close()