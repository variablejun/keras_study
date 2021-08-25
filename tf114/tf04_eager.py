import tensorflow as tf
from tensorflow.python.client.session import Session

print(tf.executing_eagerly()) #False

tf.compat.v1.disable_eager_execution()

print(tf.executing_eagerly()) #False

print(tf.__version__) #1.14.0

Hello = tf.constant('HELLO WORLD') # 상수
print(Hello) #Tensor("Const:0", shape=(), dtype=string)

sess = tf.compat.v1.Session()
print(sess.run(Hello)) #b'HELLO WORLD'