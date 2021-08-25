import tensorflow as tf # 텐서에서 변수는 반드시 초기화
sess = tf.Session()


x = tf.Variable([2], dtype= tf.float32,name='test')
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(x))