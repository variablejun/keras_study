import tensorflow as tf
from tensorflow.python.client.session import Session


model1 = tf.constant(2.0)
model2 = tf.constant(3.0)

model3 = tf.add(model1,model2)

sess = Session()

print(sess.run([model1,model2]))
print(sess.run(model3))

'''
[2.0, 3.0]
5.0
'''