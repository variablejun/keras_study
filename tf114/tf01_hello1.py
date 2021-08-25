import tensorflow as tf
print(tf.__version__) #1.14.0

Hello = tf.constant('HELLO WORLD') # 상수
print(Hello) #Tensor("Const:0", shape=(), dtype=string)

sess = tf.compat.v1.Session()
print(sess.run(Hello)) #b'HELLO WORLD'