import tensorflow as tf
from tensorflow.python.client.session import Session


model1 = tf.constant(2.0)
model2 = tf.constant(3.0)

model3 = tf.add(model1,model2)
model4 = tf.subtract(model1,model2)
model5 = tf.div(model1,model2)
model6 = tf.multiply(model1,model2)

sess = Session()

print(sess.run([model1,model2]))
print(sess.run(model3))
print(sess.run(model4))
print(sess.run(model5))
print(sess.run(model6))
'''
[2.0, 3.0]
5.0
-1.0
0.6666667
6.0
'''