import tensorflow as tf

tf.compat.v1.set_random_seed(77)
w = tf.Variable(tf.random_normal([1]), name = 'weight')

print(w) #<tf.Variable 'weight:0' shape=(1,) dtype=float32_ref>
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
aaa = sess.run(w)
print(aaa) #[1.014144]
sess.close

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
bbb = w.eval()
print(bbb) #[1.014144]
sess.close

sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())
ccc = w.eval(session=sess) # [1.014144]
print(ccc)
sess.close()