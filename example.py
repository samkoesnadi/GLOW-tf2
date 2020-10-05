import tensorflow as tf
from utils.utils import *


x = tf.random.normal((512,512), 0, 1, dtype=tf.float32)
y = tf.random.normal((512,512), 0, 0, dtype=tf.float32)

mean = tf.zeros((512,))
var = tf.ones((512,))

print(logpz(mean, var, x))
print(logpz(mean, var, y))

# a = np.array([[1,0,0,0],
#               [0,0,1,0],
#               [0,0,0,1],
#               [0,1,0,0]])
#
# print(np.linalg.det(a))
# print(np.linalg.inv(a))